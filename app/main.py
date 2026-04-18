from fastapi import FastAPI, UploadFile, Form, File, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.orm import Session
from app.database import Base, engine, SessionLocal
from app.models import Person, Attendance, UserAccount, Section, SectionStudent
from app.face_encoder import FaceEncoder
from sqlalchemy import text
import numpy as np
import pickle
import uuid
import os
import cv2
import secrets
import string
import base64
import hashlib
import hmac
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from itsdangerous import URLSafeSerializer, BadSignature
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

Instrumentator(excluded_handlers=["/metrics"]).instrument(app).expose(app, include_in_schema=False)

# Static assets (logo, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

PROFILE_UPLOAD_DIR = os.path.join("static", "uploads", "profiles")
DEFAULT_AVATAR = "/static/default-avatar.svg"
os.makedirs(PROFILE_UPLOAD_DIR, exist_ok=True)


# --- Login gate (role-based) ---
ADMIN_ID = os.getenv("FACEAPP_ADMIN_ID", "1234")
ADMIN_PASSWORD = os.getenv("FACEAPP_ADMIN_PASSWORD", "5678")
SESSION_SECRET = os.getenv("FACEAPP_SESSION_SECRET", "dev-secret-change-me")

ALLOWED_ROLES = {"admin", "faculty", "trainer", "security"}
ROLE_ORDER = ["admin", "faculty", "trainer", "security"]
AUTH_COOKIE_PREFIX = "ams_auth_"
AUTH_COOKIE_MAX_AGE = 60 * 60 * 24 * 7

# Case-sensitive captcha alphabet (avoid ambiguous chars like 0/O, 1/I/l)
CAPTCHA_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789"


def _current_role(request: Request) -> str | None:
    """Return the role from signed auth cookies."""
    for role in ROLE_ORDER:
        user = _load_auth_cookie(request, role, _profile_hint(request))
        if user:
            return role
    return None


def _profile_hint(request: Request) -> str | None:
    value = request.query_params.get("profile")
    return str(value).strip() if value else None


def _auth_serializer() -> URLSafeSerializer:
    return URLSafeSerializer(SESSION_SECRET, salt="auth")


def _cookie_key(role: str, user: str) -> str:
    safe = _b64(str(user).encode("utf-8"))
    return f"{AUTH_COOKIE_PREFIX}{role}_{safe}"


def _parse_auth_cookie(raw: str, role: str) -> str | None:
    try:
        payload = _auth_serializer().loads(raw)
    except BadSignature:
        return None
    if payload.get("role") != role:
        return None
    user = payload.get("user")
    return str(user).strip() if user else None


def _has_auth_cookie(request: Request, role: str, profile: str | None = None) -> bool:
    if not role:
        return False
    if profile:
        return _cookie_key(role, profile) in request.cookies
    prefix = f"{AUTH_COOKIE_PREFIX}{role}_"
    return any(name.startswith(prefix) for name in request.cookies.keys())


def _account_exists(role: str, username: str) -> bool:
    if role not in {"faculty", "trainer", "security"}:
        return True
    db: Session = SessionLocal()
    acct = (
        db.query(UserAccount)
        .filter(UserAccount.username == str(username).strip(), UserAccount.role == role)
        .first()
    )
    db.close()
    return acct is not None


def _load_auth_cookie(request: Request, role: str, profile: str | None = None) -> str | None:
    if not role:
        return None
    if profile:
        raw = request.cookies.get(_cookie_key(role, profile))
        if not raw:
            return None
        user = _parse_auth_cookie(raw, role)
        if user and not _account_exists(role, user):
            return None
        return user

    prefix = f"{AUTH_COOKIE_PREFIX}{role}_"
    for name in sorted(request.cookies.keys()):
        if not name.startswith(prefix):
            continue
        raw = request.cookies.get(name)
        if not raw:
            continue
        user = _parse_auth_cookie(raw, role)
        if user:
            if not _account_exists(role, user):
                continue
            return user
    return None


def _set_auth_cookie(response: RedirectResponse, role: str, user: str) -> None:
    name = _cookie_key(role, user)
    value = _auth_serializer().dumps({"role": role, "user": user})
    response.set_cookie(
        name,
        value,
        httponly=True,
        samesite="lax",
        max_age=AUTH_COOKIE_MAX_AGE,
    )


def _clear_auth_cookie(response: RedirectResponse, role: str, profile: str | None = None) -> None:
    if profile:
        response.delete_cookie(_cookie_key(role, profile))


def _is_logged_in(request: Request) -> bool:
    return _current_role(request) is not None

# Session cookie support
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    same_site="lax",
)


def _redirect_or_unauthorized(request: Request, location: str = "/login") -> None:
    accept = request.headers.get("accept", "")
    if request.method == "GET" or "text/html" in accept:
        raise HTTPException(status_code=303, headers={"Location": location})
    raise HTTPException(status_code=401, detail="Not authenticated")


def login_required(request: Request) -> str:
    role = _current_role(request)
    if role is None:
        _redirect_or_unauthorized(request, "/login")
    return role


def require_roles(*roles: str):
    allowed = {str(r).strip().lower() for r in roles}

    def _dep(request: Request) -> str:
        profile = _profile_hint(request)
        role = None
        for r in allowed:
            if profile and _has_auth_cookie(request, r, profile) and not _load_auth_cookie(request, r, profile):
                raise HTTPException(status_code=303, headers={"Location": f"/logout?role={r}&profile={profile}"})
            if _load_auth_cookie(request, r, profile):
                role = r
                break
        if role is None:
            for r in allowed:
                if profile and _has_auth_cookie(request, r, profile) and not _load_auth_cookie(request, r, profile):
                    raise HTTPException(status_code=303, headers={"Location": f"/logout?role={r}&profile={profile}"})
                if _load_auth_cookie(request, r):
                    role = r
                    break
        if role in allowed:
            return role  # for handlers that want it

        if role is None:
            _redirect_or_unauthorized(request, "/login")

        # Logged in but not allowed
        accept = request.headers.get("accept", "")
        if request.method == "GET" or "text/html" in accept:
            raise HTTPException(status_code=403, detail="Forbidden")
        raise HTTPException(status_code=403, detail="Forbidden")

    return _dep


admin_required = require_roles("admin")


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64d(text_value: str) -> bytes:
    s = str(text_value)
    pad = "=" * ((4 - (len(s) % 4)) % 4)
    return base64.urlsafe_b64decode(s + pad)


def hash_password(password: str) -> str:
    iterations = 210_000
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${_b64(salt)}${_b64(dk)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iter_s, salt_b64, hash_b64 = str(stored).split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        iterations = int(iter_s)
        salt = _b64d(salt_b64)
        expected = _b64d(hash_b64)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


def _ensure_sqlite_schema() -> None:
    # Minimal "migration" for SQLite: add missing columns when the DB already exists.
    # This avoids requiring Alembic for this project.
    Base.metadata.create_all(bind=engine)

    def has_column(conn, table: str, column: str) -> bool:
        rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
        return any(r[1] == column for r in rows)

    with engine.begin() as conn:
        # user_accounts.photo_path
        if conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='user_accounts'")).fetchone():
            if not has_column(conn, "user_accounts", "photo_path"):
                conn.execute(text("ALTER TABLE user_accounts ADD COLUMN photo_path VARCHAR"))
            if not has_column(conn, "user_accounts", "full_name"):
                conn.execute(text("ALTER TABLE user_accounts ADD COLUMN full_name VARCHAR"))

        # persons.roll_no + persons.is_blocked + persons.blocked_reason
        if conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='persons'")).fetchone():
            if not has_column(conn, "persons", "roll_no"):
                conn.execute(text("ALTER TABLE persons ADD COLUMN roll_no VARCHAR"))
            if not has_column(conn, "persons", "is_blocked"):
                conn.execute(text("ALTER TABLE persons ADD COLUMN is_blocked BOOLEAN DEFAULT 0"))
                conn.execute(text("UPDATE persons SET is_blocked = 0 WHERE is_blocked IS NULL"))
            if not has_column(conn, "persons", "blocked_reason"):
                conn.execute(text("ALTER TABLE persons ADD COLUMN blocked_reason VARCHAR"))

        # attendance.status + attendance.roll_no
        if conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='attendance'")).fetchone():
            if not has_column(conn, "attendance", "status"):
                conn.execute(text("ALTER TABLE attendance ADD COLUMN status VARCHAR DEFAULT 'marked'"))
            if not has_column(conn, "attendance", "roll_no"):
                conn.execute(text("ALTER TABLE attendance ADD COLUMN roll_no VARCHAR"))

        conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS section_students ("
                "id INTEGER PRIMARY KEY, "
                "section_id INTEGER NOT NULL, "
                "person_id INTEGER NOT NULL, "
                "UNIQUE(section_id, person_id)"
                ")"
            )
        )


_ensure_sqlite_schema()
templates = Jinja2Templates(directory="templates")

# Starlette 1.x expects TemplateResponse(request, name, context, ...).
# Keep compatibility with existing routes that call TemplateResponse(name, context, ...).
_template_response_impl = templates.TemplateResponse


def _template_response_compat(*args, **kwargs):
    if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], dict):
        name = args[0]
        context = args[1]
        request = context.get("request")
        if request is None:
            raise RuntimeError("Template context must include 'request'")
        return _template_response_impl(request, name, context, *args[2:], **kwargs)
    return _template_response_impl(*args, **kwargs)


templates.TemplateResponse = _template_response_compat
encoder = FaceEncoder()


IST = ZoneInfo("Asia/Kolkata")


def _roll_sort_key(value: str | None):
    if value is None:
        return (1, "")
    s = str(value).strip()
    if s.isdigit():
        return (0, int(s))
    return (0, s.lower())


def _fmt_ist(dt: datetime | None) -> str | None:
    if not dt:
        return None
    # DB stores naive datetimes; treat them as UTC.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST).strftime("%H:%M:%S")


def _iso_ist(dt: datetime | None) -> str | None:
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST).isoformat()


def _today_key() -> str:
    # Store local day string to match user expectations (classroom attendance)
    return datetime.now().strftime("%Y-%m-%d")


def _profile_context(request: Request, role: str | None = None, profile: str | None = None) -> dict:
    role_norm = (role or _current_role(request) or "").strip().lower()
    role_norm = role_norm if role_norm in ALLOWED_ROLES else None
    profile_hint = (profile or _profile_hint(request) or "").strip() or None
    user = _load_auth_cookie(request, role_norm, profile_hint) if role_norm else None
    display_name = str(user or (role or "User")).strip() or "User"
    photo = DEFAULT_AVATAR

    if role_norm in {"faculty", "trainer", "security"} and user:
        db: Session = SessionLocal()
        acct = (
            db.query(UserAccount)
            .filter(UserAccount.username == str(user).strip(), UserAccount.role == role_norm)
            .first()
        )
        if acct and acct.photo_path:
            photo = acct.photo_path
            display_name = acct.full_name or acct.username
        db.close()

    return {
        "user_name": display_name,
        "user_role": role_norm or "user",
        "user_photo": photo,
        "profile": user or "",
    }


def _load_known_embeddings(db: Session, allowed_ids: set[int] | None = None):
    if allowed_ids:
        persons = db.query(Person).filter(Person.id.in_(list(allowed_ids))).all()
    else:
        persons = db.query(Person).all()
    if not persons:
        return np.empty((0,)), []

    embeddings = []
    people = []
    for p in persons:
        emb = pickle.loads(p.embedding)
        emb = encoder.l2_normalize(np.array(emb))
        embeddings.append(emb)
        people.append(
            {
                "id": p.id,
                "name": p.name,
                "roll_no": p.roll_no,
                "is_blocked": bool(p.is_blocked),
                "blocked_reason": p.blocked_reason,
            }
        )

    return np.vstack(embeddings), people


def _check_duplicate_person(db: Session, person_id: int, roll_no: str) -> str | None:
    existing_id = db.query(Person).filter(Person.id == person_id).first()
    if existing_id:
        return f"Regn no {person_id} already exists"

    normalized_roll = (roll_no or "").strip()
    if normalized_roll:
        existing_roll = db.query(Person).filter(Person.roll_no == normalized_roll).first()
        if existing_roll:
            return f"Roll no {normalized_roll} already exists"

    return None


def _faculty_profile(request: Request) -> str | None:
    profile = _profile_hint(request)
    if profile and _load_auth_cookie(request, "faculty", profile):
        return profile
    return _load_auth_cookie(request, "faculty")


def _faculty_section_ids(db: Session, request: Request) -> set[int]:
    profile = _faculty_profile(request)
    if not profile:
        return set()
    rows = db.query(Section).filter(Section.faculty_uid == profile).all()
    return {int(r.id) for r in rows}


def _allowed_person_ids(db: Session, request: Request, role: str, section_id: int | None = None) -> set[int] | None:
    if role != "faculty":
        return None
    section_ids = _faculty_section_ids(db, request)
    if not section_ids:
        return set()
    if section_id is None or int(section_id) not in section_ids:
        return set()
    rows = (
        db.query(SectionStudent.person_id)
        .filter(SectionStudent.section_id == int(section_id))
        .distinct()
        .all()
    )
    return {int(r[0]) for r in rows}


def _people_for_role(db: Session, request: Request, role: str, section_id: int | None = None) -> list[Person]:
    allowed_ids = _allowed_person_ids(db, request, role, section_id)
    if allowed_ids is None:
        return db.query(Person).all()
    if not allowed_ids:
        return []
    return db.query(Person).filter(Person.id.in_(list(allowed_ids))).all()


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, msg: str | None = None):
    captcha_code = "".join(secrets.choice(CAPTCHA_ALPHABET) for _ in range(6))
    request.session["login_captcha"] = captcha_code
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "error": None,
            "captcha_code": captcha_code,
            "msg": msg,
        },
    )


@app.post("/login", response_class=HTMLResponse)
async def login_submit(
    request: Request,
    role: str = Form(...),
    user_id: str = Form(...),
    password: str = Form(...),
    captcha: str = Form(""),
):
    role_norm = (role or "").strip().lower()

    if role_norm not in ALLOWED_ROLES:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": "Invalid role",
                "captcha_code": request.session.get("login_captcha"),
            },
            status_code=400,
        )

    expected_raw = str(request.session.get("login_captcha") or "").strip()
    provided_raw = str(captcha or "").strip()
    # Users may type spaces due to UI letter-spacing; ignore spaces only.
    expected = expected_raw.replace(" ", "")
    provided = provided_raw.replace(" ", "")

    if not expected:
        # Session/cookie mismatch or expired session data
        new_code = "".join(secrets.choice(CAPTCHA_ALPHABET) for _ in range(6))
        request.session["login_captcha"] = new_code
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": "Session expired. Please try again.",
                "captcha_code": new_code,
            },
            status_code=400,
        )

    # Case-sensitive match
    if provided != expected:
        # Regenerate captcha to prevent brute force
        new_code = "".join(secrets.choice(CAPTCHA_ALPHABET) for _ in range(6))
        request.session["login_captcha"] = new_code
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid captcha", "captcha_code": new_code},
            status_code=400,
        )

    user_id_norm = str(user_id).strip()
    password_norm = str(password).strip()

    if role_norm == "admin":
        if user_id_norm == ADMIN_ID and password_norm == ADMIN_PASSWORD:
            response = RedirectResponse(url="/", status_code=303)
            _set_auth_cookie(response, "admin", user_id_norm)
            return response

    if role_norm in {"faculty", "trainer", "security"}:
        db: Session = SessionLocal()
        acct = (
            db.query(UserAccount)
            .filter(UserAccount.username == user_id_norm, UserAccount.role == role_norm)
            .first()
        )
        db.close()

        if acct and verify_password(password_norm, acct.password_hash):
            profile_q = f"?profile={user_id_norm}"
            if role_norm == "faculty":
                response = RedirectResponse(url=f"/faculty{profile_q}", status_code=303)
            elif role_norm == "trainer":
                response = RedirectResponse(url=f"/trainer{profile_q}", status_code=303)
            else:
                response = RedirectResponse(url=f"/security{profile_q}", status_code=303)
            _set_auth_cookie(response, role_norm, user_id_norm)
            return response

    # Keep captcha visible on wrong password (and rotate it)
    new_code = "".join(secrets.choice(CAPTCHA_ALPHABET) for _ in range(6))
    request.session["login_captcha"] = new_code
    err = "Invalid credentials"
    if role_norm == "admin":
        err = "Invalid Admin ID or password"
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": err, "captcha_code": new_code},
        status_code=401,
    )


@app.get("/logout")
async def logout(request: Request, role: str | None = None, profile: str | None = None):
    role_norm = (role or "").strip().lower()
    profile_norm = (profile or "").strip()
    response = RedirectResponse(url="/login", status_code=303)
    if role_norm in ALLOWED_ROLES:
        if profile_norm:
            _clear_auth_cookie(response, role_norm, profile_norm)
        else:
            prefix = f"{AUTH_COOKIE_PREFIX}{role_norm}_"
            for name in request.cookies.keys():
                if name.startswith(prefix):
                    response.delete_cookie(name)
    else:
        for r in ALLOWED_ROLES:
            prefix = f"{AUTH_COOKIE_PREFIX}{r}_"
            for name in request.cookies.keys():
                if name.startswith(prefix):
                    response.delete_cookie(name)
    return response


@app.get("/auth/ping")
async def auth_ping(request: Request, role: str | None = None, profile: str | None = None):
    role_norm = (role or "").strip().lower()
    if role_norm not in ALLOWED_ROLES:
        return {"ok": False}
    user = _load_auth_cookie(request, role_norm, (profile or "").strip() or None)
    return {"ok": bool(user)}


@app.get(
    "/change-password",
    response_class=HTMLResponse,
    dependencies=[Depends(require_roles("faculty", "trainer", "security"))],
)
async def change_password_page(request: Request, role: str | None = None, profile: str | None = None):
    role_norm = (role or "").strip().lower()
    role_final = role_norm if role_norm in ALLOWED_ROLES else _current_role(request)
    ctx = {"request": request, "error": None, "success": None, **_profile_context(request, role_final, profile)}
    return templates.TemplateResponse("change_password.html", ctx)


@app.post(
    "/change-password",
    response_class=HTMLResponse,
    dependencies=[Depends(require_roles("faculty", "trainer", "security"))],
)
async def change_password_submit(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
    role: str | None = Form(None),
    profile: str | None = Form(None),
):
    role_norm = (role or "").strip().lower()
    role_final = role_norm if role_norm in ALLOWED_ROLES else _current_role(request)
    user = _load_auth_cookie(request, role_final, profile) if role_final else None

    if not role_final or not user:
        return RedirectResponse(url="/login", status_code=303)

    current_password = str(current_password or "").strip()
    new_password = str(new_password or "").strip()
    confirm_password = str(confirm_password or "").strip()

    if not current_password or not new_password:
        ctx = {"request": request, "error": "All fields are required", "success": None, **_profile_context(request, role_final, profile)}
        return templates.TemplateResponse("change_password.html", ctx, status_code=400)

    if new_password != confirm_password:
        ctx = {"request": request, "error": "New password does not match", "success": None, **_profile_context(request, role_final, profile)}
        return templates.TemplateResponse("change_password.html", ctx, status_code=400)

    if len(new_password) < 6:
        ctx = {"request": request, "error": "Password must be at least 6 characters", "success": None, **_profile_context(request, role_final, profile)}
        return templates.TemplateResponse("change_password.html", ctx, status_code=400)

    db: Session = SessionLocal()
    acct = (
        db.query(UserAccount)
        .filter(UserAccount.username == str(user).strip(), UserAccount.role == role_final)
        .first()
    )
    if not acct or not verify_password(current_password, acct.password_hash):
        db.close()
        ctx = {"request": request, "error": "Current password is incorrect", "success": None, **_profile_context(request, role_final, profile)}
        return templates.TemplateResponse("change_password.html", ctx, status_code=400)

    acct.password_hash = hash_password(new_password)
    db.commit()
    db.close()
    msg = "Password changed successfully. Logging out."
    response = RedirectResponse(url=f"/login?msg={msg.replace(' ', '+')}", status_code=303)
    if role_final:
        if profile:
            _clear_auth_cookie(response, role_final, profile)
        else:
            prefix = f"{AUTH_COOKIE_PREFIX}{role_final}_"
            for name in request.cookies.keys():
                if name.startswith(prefix):
                    response.delete_cookie(name)
    return response

@app.get("/", response_class=HTMLResponse, dependencies=[Depends(login_required)])
async def home(request: Request, role: str = Depends(login_required)):
    if role == "admin":
        return templates.TemplateResponse("home.html", {"request": request})
    if role == "faculty":
        return RedirectResponse(url="/faculty", status_code=303)
    if role == "trainer":
        return RedirectResponse(url="/trainer", status_code=303)
    if role == "security":
        return RedirectResponse(url="/security", status_code=303)
    return RedirectResponse(url="/login", status_code=303)


@app.get(
    "/faculty",
    response_class=HTMLResponse,
    dependencies=[Depends(require_roles("admin", "faculty"))],
)
async def faculty_dashboard(request: Request):
    db: Session = SessionLocal()
    profile = _profile_hint(request)
    faculty_uid = profile or _load_auth_cookie(request, "faculty")
    sections = []
    if faculty_uid:
        rows = db.query(Section).filter(Section.faculty_uid == faculty_uid).order_by(Section.course_code.asc()).all()
        counts = dict(
            db.query(SectionStudent.section_id, text("COUNT(1)")).group_by(SectionStudent.section_id).all()
        )
        sections = [
            {
                "id": r.id,
                "course_code": r.course_code,
                "course_name": r.course_name,
                "count": int(counts.get(r.id, 0)),
            }
            for r in rows
        ]
    db.close()
    ctx = {
        "request": request,
        "sections": sections,
        **_profile_context(request, "faculty", _profile_hint(request)),
    }
    return templates.TemplateResponse("faculty_home.html", ctx)


@app.get(
    "/faculty/section",
    response_class=HTMLResponse,
    dependencies=[Depends(require_roles("admin", "faculty"))],
)
async def faculty_section_page(
    request: Request,
    section_id: int | None = None,
    role: str = Depends(require_roles("admin", "faculty")),
):
    if section_id is None:
        return RedirectResponse(url=f"/faculty?profile={_profile_hint(request) or ''}", status_code=303)

    db: Session = SessionLocal()
    section = db.query(Section).filter(Section.id == section_id).first()
    allowed_ids = _allowed_person_ids(db, request, "faculty", section_id) if role == "faculty" else None
    db.close()
    if not section or (allowed_ids is not None and not allowed_ids):
        return RedirectResponse(url=f"/faculty?profile={_profile_hint(request) or ''}", status_code=303)

    ctx = {
        "request": request,
        "section": section,
        **_profile_context(request, "faculty", _profile_hint(request)),
    }
    return templates.TemplateResponse("faculty_section.html", ctx)


@app.get(
    "/trainer",
    response_class=HTMLResponse,
    dependencies=[Depends(require_roles("admin", "trainer"))],
)
async def trainer_dashboard(request: Request):
    ctx = {"request": request, **_profile_context(request, "trainer", _profile_hint(request))}
    return templates.TemplateResponse("trainer_home.html", ctx)


@app.get(
    "/security",
    response_class=HTMLResponse,
    dependencies=[Depends(require_roles("admin", "security"))],
)
async def security_dashboard(request: Request):
    ctx = {"request": request, **_profile_context(request, "security", _profile_hint(request))}
    return templates.TemplateResponse("security_home.html", ctx)


@app.get(
    "/security/students",
    response_class=HTMLResponse,
    dependencies=[Depends(require_roles("admin", "security"))],
)
async def security_students_page(request: Request, msg: str | None = None):
    db: Session = SessionLocal()
    people = _people_for_role(db, request, "security")
    people = sorted(people, key=lambda p: _roll_sort_key(p.roll_no))
    db.close()
    ctx = {
        "request": request,
        "people": people,
        "count": len(people),
        "msg": msg,
        **_profile_context(request, "security", _profile_hint(request)),
    }
    return templates.TemplateResponse("security_students.html", ctx)


@app.post("/security/block", dependencies=[Depends(require_roles("admin", "security"))])
async def security_block_student(person_id: int = Form(...), reason: str = Form(...)):
    reason_norm = str(reason or "").strip()
    if not reason_norm:
        return RedirectResponse(url="/security/students?msg=Block+reason+required", status_code=303)

    db: Session = SessionLocal()
    person = db.query(Person).filter(Person.id == person_id).first()
    if not person:
        db.close()
        return RedirectResponse(url="/security/students?msg=Student+not+found", status_code=303)

    person.is_blocked = True
    person.blocked_reason = reason_norm
    today = _today_key()
    db.query(Attendance).filter(
        Attendance.person_id == person.id,
        Attendance.day == today,
    ).update({Attendance.status: "unmarked"})
    db.commit()
    db.close()
    return RedirectResponse(url="/security/students?msg=Student+blocked", status_code=303)


@app.post("/security/unblock", dependencies=[Depends(require_roles("admin", "security"))])
async def security_unblock_student(person_id: int = Form(...)):
    db: Session = SessionLocal()
    person = db.query(Person).filter(Person.id == person_id).first()
    if not person:
        db.close()
        return RedirectResponse(url="/security/students?msg=Student+not+found", status_code=303)

    person.is_blocked = False
    person.blocked_reason = None
    db.commit()
    db.close()
    return RedirectResponse(url="/security/students?msg=Student+unblocked", status_code=303)


@app.get(
    "/train",
    response_class=HTMLResponse,
    dependencies=[Depends(require_roles("admin", "trainer"))],
    include_in_schema=False,
)
async def train_page(request: Request, role: str = Depends(require_roles("admin", "trainer"))):
    ctx = {"request": request, **_profile_context(request, role, _profile_hint(request))}
    return templates.TemplateResponse("train.html", ctx)


@app.get(
    "/train-live",
    response_class=HTMLResponse,
    dependencies=[Depends(require_roles("admin", "trainer"))],
    include_in_schema=False,
)
async def train_live_page(request: Request, role: str = Depends(require_roles("admin", "trainer"))):
    ctx = {"request": request, **_profile_context(request, role, _profile_hint(request))}
    return templates.TemplateResponse("train_live.html", ctx)


@app.get("/live", response_class=HTMLResponse, dependencies=[Depends(require_roles("admin", "faculty"))])
async def live_page(
    request: Request,
    section_id: int | None = None,
    role: str = Depends(require_roles("admin", "faculty")),
):
    if role == "faculty":
        db: Session = SessionLocal()
        allowed_ids = _allowed_person_ids(db, request, role, section_id)
        db.close()
        if not allowed_ids:
            return RedirectResponse(url=f"/faculty?profile={_profile_hint(request) or ''}", status_code=303)
    ctx = {
        "request": request,
        "section_id": section_id,
        **_profile_context(request, role, _profile_hint(request)),
    }
    return templates.TemplateResponse("live.html", ctx)


@app.get("/admin", response_class=HTMLResponse, dependencies=[Depends(admin_required)])
async def admin_page(request: Request, msg: str | None = None):
    db: Session = SessionLocal()
    people = db.query(Person).all()
    people = sorted(people, key=lambda p: _roll_sort_key(p.roll_no))
    accounts = db.query(UserAccount).order_by(UserAccount.role.asc(), UserAccount.username.asc()).all()
    sections = db.query(Section).order_by(Section.course_code.asc()).all()
    counts = dict(
        db.query(SectionStudent.section_id, text("COUNT(1)")).group_by(SectionStudent.section_id).all()
    )
    account_by_uid = {a.username: a for a in accounts}
    section_rows = []
    for s in sections:
        faculty = account_by_uid.get(s.faculty_uid)
        section_rows.append(
            {
                "id": s.id,
                "course_code": s.course_code,
                "course_name": s.course_name,
                "faculty_uid": s.faculty_uid,
                "faculty_name": faculty.full_name if faculty else None,
                "count": int(counts.get(s.id, 0)),
            }
        )
    db.close()
    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "people": people,
            "count": len(people),
            "accounts": accounts,
            "sections": section_rows,
            "msg": msg,
        },
    )


@app.get("/admin/api/students", dependencies=[Depends(admin_required)])
async def admin_students_api():
    db: Session = SessionLocal()
    people = db.query(Person).all()
    people = sorted(people, key=lambda p: _roll_sort_key(p.roll_no))
    rows = [{"id": p.id, "roll_no": p.roll_no, "name": p.name} for p in people]
    db.close()
    return {"count": len(rows), "people": rows}


@app.post("/admin/add-student", dependencies=[Depends(admin_required)])
async def admin_add_student(
    person_id: int = Form(...),
    roll_no: str = Form(...),
    name: str = Form(...),
    files: list[UploadFile] = File(...),
):
    # Reuse the same training logic
    db: Session = SessionLocal()
    duplicate_error = _check_duplicate_person(db, person_id, roll_no)
    if duplicate_error:
        db.close()
        return {"error": duplicate_error}

    embeddings = []

    for file in files:
        contents = await file.read()
        emb = encoder.encode_image(contents)
        if emb is not None:
            embeddings.append(emb)

    if not embeddings:
        db.close()
        return {"error": f"No valid faces found for {name}"}

    avg_embedding = np.mean(embeddings, axis=0)
    avg_embedding = encoder.l2_normalize(avg_embedding)
    serialized = pickle.dumps(avg_embedding)

    person = Person(id=person_id, roll_no=roll_no.strip(), name=name, embedding=serialized)
    db.add(person)
    db.commit()
    db.close()
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/delete-student", dependencies=[Depends(admin_required)])
async def admin_delete_student(person_id: int = Form(...)):
    db: Session = SessionLocal()
    # delete attendance rows for that student
    db.query(Attendance).filter(Attendance.person_id == person_id).delete()
    # delete person
    db.query(Person).filter(Person.id == person_id).delete()
    db.commit()
    db.close()
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/create-account", dependencies=[Depends(admin_required)])
async def admin_create_account(
    role: str = Form(...),
    full_name: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    photo: UploadFile = File(...),
):
    role_norm = (role or "").strip().lower()
    full_name_norm = (full_name or "").strip()
    username_norm = (username or "").strip()
    password_norm = (password or "").strip()

    if role_norm not in {"faculty", "trainer", "security"}:
        return RedirectResponse(url="/admin?msg=Invalid+role", status_code=303)
    if not full_name_norm or not username_norm or not password_norm:
        return RedirectResponse(url="/admin?msg=Name,+user+id+and+password+required", status_code=303)

    if not photo or not (photo.filename or "").strip():
        return RedirectResponse(url="/admin?msg=Profile+photo+required", status_code=303)
    if not (photo.content_type or "").startswith("image/"):
        return RedirectResponse(url="/admin?msg=Invalid+photo+type", status_code=303)

    contents = await photo.read()
    if not contents:
        return RedirectResponse(url="/admin?msg=Empty+photo+file", status_code=303)

    ext = os.path.splitext(photo.filename or "")[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        ext = ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(PROFILE_UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(contents)
    photo_url = f"/static/uploads/profiles/{filename}"

    db: Session = SessionLocal()
    existing = db.query(UserAccount).filter(UserAccount.username == username_norm).first()
    if existing:
        db.close()
        return RedirectResponse(url="/admin?msg=Username+already+exists", status_code=303)

    acct = UserAccount(
        role=role_norm,
        username=username_norm,
        full_name=full_name_norm,
        password_hash=hash_password(password_norm),
        photo_path=photo_url,
    )
    db.add(acct)
    db.commit()
    db.close()
    return RedirectResponse(url="/admin?msg=Account+created", status_code=303)


@app.post("/admin/create-section", dependencies=[Depends(admin_required)])
async def admin_create_section(
    course_code: str = Form(...),
    course_name: str = Form(...),
    faculty_uid: str = Form(...),
    student_ids: list[int] = Form([]),
):
    code = (course_code or "").strip()
    name = (course_name or "").strip()
    faculty = (faculty_uid or "").strip()
    if not code or not name or not faculty:
        return RedirectResponse(url="/admin?msg=Course+code,+name+and+faculty+required", status_code=303)

    db: Session = SessionLocal()
    acct = db.query(UserAccount).filter(UserAccount.username == faculty, UserAccount.role == "faculty").first()
    if not acct:
        db.close()
        return RedirectResponse(url="/admin?msg=Faculty+UID+not+found", status_code=303)

    section = Section(course_code=code, course_name=name, faculty_uid=faculty)
    db.add(section)
    db.commit()
    db.refresh(section)

    if student_ids:
        existing = db.query(SectionStudent.person_id).filter(
            SectionStudent.section_id == section.id,
            SectionStudent.person_id.in_(student_ids),
        ).all()
        existing_ids = {int(r[0]) for r in existing}
        new_links = [
            SectionStudent(section_id=section.id, person_id=int(pid))
            for pid in student_ids
            if int(pid) not in existing_ids
        ]
        if new_links:
            db.add_all(new_links)
        db.commit()

    db.close()
    return RedirectResponse(url="/admin?msg=Section+created", status_code=303)


@app.post("/admin/delete-section", dependencies=[Depends(admin_required)])
async def admin_delete_section(section_id: int = Form(...)):
    db: Session = SessionLocal()
    db.query(SectionStudent).filter(SectionStudent.section_id == section_id).delete()
    db.query(Section).filter(Section.id == section_id).delete()
    db.commit()
    db.close()
    return RedirectResponse(url="/admin?msg=Section+deleted", status_code=303)


@app.post("/admin/update-section-faculty", dependencies=[Depends(admin_required)])
async def admin_update_section_faculty(section_id: int = Form(...), faculty_uid: str = Form(...)):
    faculty = (faculty_uid or "").strip()
    if not faculty:
        return RedirectResponse(url="/admin?msg=Faculty+UID+required", status_code=303)

    db: Session = SessionLocal()
    acct = db.query(UserAccount).filter(UserAccount.username == faculty, UserAccount.role == "faculty").first()
    if not acct:
        db.close()
        return RedirectResponse(url="/admin?msg=Faculty+UID+not+found", status_code=303)

    section = db.query(Section).filter(Section.id == section_id).first()
    if not section:
        db.close()
        return RedirectResponse(url="/admin?msg=Section+not+found", status_code=303)

    section.faculty_uid = faculty
    db.commit()
    db.close()
    return RedirectResponse(url="/admin?msg=Section+faculty+updated", status_code=303)


@app.post("/admin/delete-account", dependencies=[Depends(admin_required)])
async def admin_delete_account(account_id: int = Form(...)):
    db: Session = SessionLocal()
    db.query(UserAccount).filter(UserAccount.id == account_id).delete()
    db.commit()
    db.close()
    return RedirectResponse(url="/admin?msg=Account+deleted", status_code=303)


@app.get("/attendance", response_class=HTMLResponse, dependencies=[Depends(require_roles("admin", "faculty"))])
async def attendance_page(
    request: Request,
    day: str | None = None,
    section_id: int | None = None,
    role: str = Depends(require_roles("admin", "faculty")),
):
    selected_day = day or _today_key()
    db: Session = SessionLocal()
    if role == "faculty":
        allowed_ids = _allowed_person_ids(db, request, role, section_id)
        if not allowed_ids:
            db.close()
            return RedirectResponse(url=f"/faculty?profile={_profile_hint(request) or ''}", status_code=303)
    people = _people_for_role(db, request, role, section_id)
    people = sorted(people, key=lambda p: _roll_sort_key(p.roll_no))

    att_rows = (
        db.query(Attendance)
        .filter(Attendance.day == selected_day)
        .order_by(Attendance.marked_at.desc())
        .all()
    )
    att_by_person: dict[int, Attendance] = {int(a.person_id): a for a in att_rows}

    view_rows = []
    for p in people:
        a = att_by_person.get(int(p.id))
        raw_status = (a.status if a else None) or "unmarked"
        blocked = bool(p.is_blocked)
        present = bool(a) and raw_status == "marked" and not blocked
        status_label = "blocked" if blocked else ("present" if present else "absent")
        view_rows.append(
            {
                "attendance_id": a.id if a else None,
                "person_id": p.id,
                "roll_no": p.roll_no,
                "name": p.name,
                "present": present,
                "status": status_label,
                "blocked": blocked,
                "block_reason": p.blocked_reason,
                # Show the time the attendance was actually marked (fixed).
                "marked_time_ist": _fmt_ist(a.marked_at) if present else None,
            }
        )

    marked_count = sum(1 for r in view_rows if r["present"]) 
    db.close()
    return templates.TemplateResponse(
        "attendance.html",
        {
            "request": request,
            "day": selected_day,
            "rows": view_rows,
            "count": marked_count,
            "section_id": section_id,
            **_profile_context(request, role, _profile_hint(request)),
        },
    )


@app.get("/attendance/api", dependencies=[Depends(require_roles("admin", "faculty"))])
async def attendance_api(
    request: Request,
    day: str | None = None,
    section_id: int | None = None,
    role: str = Depends(require_roles("admin", "faculty")),
):
    selected_day = day or _today_key()
    db: Session = SessionLocal()
    if role == "faculty":
        allowed_ids = _allowed_person_ids(db, request, role, section_id)
        if not allowed_ids:
            db.close()
            return {"day": selected_day, "count": 0, "rows": []}
    people = _people_for_role(db, request, role, section_id)
    people = sorted(people, key=lambda p: _roll_sort_key(p.roll_no))

    att_rows = (
        db.query(Attendance)
        .filter(Attendance.day == selected_day)
        .order_by(Attendance.marked_at.desc())
        .all()
    )
    att_by_person: dict[int, Attendance] = {int(a.person_id): a for a in att_rows}

    view_rows = []
    for p in people:
        a = att_by_person.get(int(p.id))
        raw_status = (a.status if a else None) or "unmarked"
        blocked = bool(p.is_blocked)
        present = bool(a) and raw_status == "marked" and not blocked
        status_label = "blocked" if blocked else ("present" if present else "absent")
        view_rows.append(
            {
                "attendance_id": a.id if a else None,
                "person_id": p.id,
                "roll_no": p.roll_no,
                "name": p.name,
                "present": present,
                "status": status_label,
                "blocked": blocked,
                "block_reason": p.blocked_reason,
                "marked_time_ist": _fmt_ist(a.marked_at) if present else None,
            }
        )

    marked_count = sum(1 for r in view_rows if r["present"])
    db.close()
    return {"day": selected_day, "count": marked_count, "rows": view_rows}


@app.get("/attendance/api/today", dependencies=[Depends(require_roles("admin", "faculty"))])
async def attendance_today_api(
    request: Request,
    section_id: int | None = None,
    role: str = Depends(require_roles("admin", "faculty")),
):
    day = _today_key()
    db: Session = SessionLocal()
    allowed_ids = _allowed_person_ids(db, request, role, section_id)
    query = db.query(Attendance).filter(Attendance.day == day)
    if allowed_ids is not None:
        if not allowed_ids:
            db.close()
            return {"day": day, "count": 0, "marked": []}
        query = query.filter(Attendance.person_id.in_(list(allowed_ids)))
    rows = query.order_by(Attendance.marked_at.desc()).all()
    db.close()
    marked_count = sum(1 for r in rows if (r.status or "marked") == "marked")
    return {
        "day": day,
        "count": marked_count,
        "marked": [
            {
                "attendance_id": r.id,
                "person_id": r.person_id,
                "roll_no": r.roll_no,
                "name": r.name,
                "marked_at": r.marked_at.isoformat() if r.marked_at else None,
                "marked_at_ist": _iso_ist(r.marked_at),
                "status": r.status,
            }
            for r in rows
        ],
    }
@app.post(
    "/train/",
    dependencies=[Depends(require_roles("admin", "trainer"))],
    include_in_schema=False,
)
async def train_person(
    person_id: int = Form(...),
    roll_no: str = Form(...),
    name: str = Form(...),
    files: list[UploadFile] = File(...)
):
    db: Session = SessionLocal()
    duplicate_error = _check_duplicate_person(db, person_id, roll_no)
    if duplicate_error:
        db.close()
        return {"error": duplicate_error}
    
    embeddings = []

    for file in files:
        contents = await file.read()
        emb = encoder.encode_image(contents)
        if emb is not None:
            embeddings.append(emb)

    if not embeddings:
        return {"error": f"No valid faces found for {name}"}

    avg_embedding = np.mean(embeddings, axis=0)
    avg_embedding = encoder.l2_normalize(avg_embedding)
    serialized = pickle.dumps(avg_embedding)

    person = Person(id=person_id, roll_no=roll_no.strip(), name=name, embedding=serialized)
    db.add(person)
    db.commit()
    db.close()

    return {"message": f"✅ {name} (ID={person_id}) trained successfully!"}

@app.get("/recognize-page", response_class=HTMLResponse, dependencies=[Depends(admin_required)])
async def recognize_page(request: Request):
    return templates.TemplateResponse("recognize.html", {"request": request})


@app.get(
    "/recognize-live",
    response_class=HTMLResponse,
    dependencies=[Depends(admin_required)],
    include_in_schema=False,
)
async def recognize_live_page(request: Request):
    return templates.TemplateResponse("recognize_live.html", {"request": request})

@app.post("/recognize/", dependencies=[Depends(admin_required)])
async def recognize_faces(file: UploadFile = File(...)):
    db: Session = SessionLocal()
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    known_embeddings, known = _load_known_embeddings(db)
    db.close()

    # Detect faces in the uploaded image
    faces = encoder.app.get(img)
    threshold_cos = 0.4  # tweak for sensitivity

    for face in faces:
        emb = encoder.l2_normalize(face.embedding)
        name = "Unknown"
        if known_embeddings.size != 0:
            cos_sim = np.dot(known_embeddings, emb)
            idx = int(np.argmax(cos_sim))
            max_sim = float(cos_sim[idx])
            if max_sim > threshold_cos:
                name = known[idx]["name"]

        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{name}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Save result image temporarily
    output_path = f"recognized_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(output_path, img)

    return FileResponse(output_path, media_type="image/jpeg")


@app.post("/recognize/live-frame", dependencies=[Depends(admin_required)])
async def recognize_live_frame(file: UploadFile = File(...)):
    db: Session = SessionLocal()

    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        db.close()
        return {"error": "Could not decode image"}

    known_embeddings, known = _load_known_embeddings(db)
    db.close()

    faces = encoder.app.get(img)
    threshold_cos = 0.4

    in_frame: list[dict] = []
    unknown_count = 0

    for face in faces:
        if known_embeddings.size == 0:
            unknown_count += 1
            continue

        emb = encoder.l2_normalize(face.embedding)
        cos_sim = np.dot(known_embeddings, emb)
        idx = int(np.argmax(cos_sim))
        max_sim = float(cos_sim[idx])

        if max_sim <= threshold_cos:
            unknown_count += 1
            continue

        in_frame.append(
            {
                "person_id": int(known[idx]["id"]),
                "roll_no": known[idx].get("roll_no"),
                "name": known[idx]["name"],
                "similarity": max_sim,
            }
        )

    # Deduplicate by person_id (multiple faces can match same person in noisy frames)
    dedup: dict[int, dict] = {}
    for row in in_frame:
        pid = int(row["person_id"])
        prev = dedup.get(pid)
        if prev is None or float(row.get("similarity", 0)) > float(prev.get("similarity", 0)):
            dedup[pid] = row

    final_rows = list(dedup.values())
    final_rows.sort(key=lambda r: float(r.get("similarity", 0.0)), reverse=True)

    return {
        "faces_detected": len(faces),
        "unknown_count": unknown_count,
        "in_frame": final_rows,
    }


@app.post("/attendance/mark", dependencies=[Depends(require_roles("admin", "faculty"))])
async def mark_attendance(
    request: Request,
    role: str = Depends(require_roles("admin", "faculty")),
    file: UploadFile = File(...),
):
    day = _today_key()
    db: Session = SessionLocal()
    section_id = request.query_params.get("section_id")
    section_id_val = int(section_id) if section_id and str(section_id).isdigit() else None

    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        db.close()
        return {"error": "Could not decode image"}

    allowed_ids = _allowed_person_ids(db, request, role, section_id_val)
    if allowed_ids is not None and not allowed_ids:
        db.close()
        return {"error": "No students assigned to this faculty"}

    known_embeddings, known = _load_known_embeddings(db, allowed_ids)
    faces = encoder.app.get(img)

    threshold_cos = 0.4
    newly_marked: list[dict] = []
    already_marked: list[dict] = []
    blocked_marked: list[dict] = []
    unknown_count = 0
    in_frame_person_ids: set[int] = set()

    for face in faces:
        emb = encoder.l2_normalize(face.embedding)

        if known_embeddings.size == 0:
            unknown_count += 1
            continue

        cos_sim = np.dot(known_embeddings, emb)
        idx = int(np.argmax(cos_sim))
        max_sim = float(cos_sim[idx])

        if max_sim <= threshold_cos:
            unknown_count += 1
            continue

        person_id = int(known[idx]["id"])
        name = known[idx]["name"]
        roll_no = known[idx].get("roll_no")
        blocked = bool(known[idx].get("is_blocked"))
        if blocked:
            blocked_marked.append(
                {
                    "person_id": person_id,
                    "roll_no": roll_no,
                    "name": name,
                    "reason": known[idx].get("blocked_reason"),
                    "similarity": max_sim,
                }
            )
            continue
        existing = (
            db.query(Attendance)
            .filter(Attendance.person_id == person_id, Attendance.day == day)
            .first()
        )

        if existing:
            in_frame_person_ids.add(person_id)
            if existing.status != "marked":
                existing.status = "marked"
                existing.marked_at = datetime.utcnow()
                existing.roll_no = roll_no
                existing.name = name
                db.commit()
            already_marked.append(
                {
                    "person_id": person_id,
                    "roll_no": roll_no,
                    "name": name,
                    "marked_at": existing.marked_at.isoformat() if existing.marked_at else None,
                    "similarity": max_sim,
                    "status": existing.status,
                }
            )
            continue

        row = Attendance(person_id=person_id, roll_no=roll_no, name=name, day=day, status="marked")
        db.add(row)
        try:
            db.commit()
        except Exception:
            db.rollback()
            existing = (
                db.query(Attendance)
                .filter(Attendance.person_id == person_id, Attendance.day == day)
                .first()
            )
            already_marked.append(
                {
                    "person_id": person_id,
                    "roll_no": roll_no,
                    "name": name,
                    "marked_at": existing.marked_at.isoformat() if existing and existing.marked_at else None,
                    "similarity": max_sim,
                    "status": existing.status if existing else None,
                }
            )
        else:
            db.refresh(row)
            in_frame_person_ids.add(person_id)
            newly_marked.append(
                {
                    "person_id": person_id,
                    "roll_no": roll_no,
                    "name": name,
                    "marked_at": row.marked_at.isoformat() if row.marked_at else None,
                    "marked_at_ist": _iso_ist(row.marked_at),
                    "similarity": max_sim,
                    "status": row.status,
                }
            )

    # Return only the people seen in THIS frame for the live UI
    in_frame_rows = []
    if in_frame_person_ids:
        in_frame_rows = (
            db.query(Attendance)
            .filter(Attendance.day == day, Attendance.person_id.in_(list(in_frame_person_ids)))
            .order_by(Attendance.marked_at.desc())
            .all()
        )
    db.close()

    return {
        "day": day,
        "faces_detected": len(faces),
        "newly_marked": newly_marked,
        "already_marked": already_marked,
        "blocked": blocked_marked,
        "blocked_count": len(blocked_marked),
        "unknown_count": unknown_count,
        "in_frame": [
            {
                "attendance_id": r.id,
                "person_id": r.person_id,
                "roll_no": r.roll_no,
                "name": r.name,
                "marked_at": r.marked_at.isoformat() if r.marked_at else None,
                "marked_at_ist": _iso_ist(r.marked_at),
                "status": (r.status or "marked"),
            }
            for r in in_frame_rows
        ],
    }


@app.post("/attendance/set-status", dependencies=[Depends(require_roles("admin", "faculty"))])
async def attendance_set_status(
    request: Request,
    role: str = Depends(require_roles("admin", "faculty")),
    status: str = Form(...),
    day: str | None = Form(None),
    section_id: int | None = Form(None),
    attendance_id: int | None = Form(None),
    person_id: int | None = Form(None),
):
    # status: marked | unmarked
    if status not in {"marked", "unmarked"}:
        return {"error": "Invalid status"}

    selected_day = day or _today_key()
    qs = [f"day={selected_day}"]
    if section_id is not None:
        qs.append(f"section_id={int(section_id)}")
    redirect_url = f"/attendance?{'&'.join(qs)}"
    db: Session = SessionLocal()

    row = None
    if attendance_id is not None:
        row = db.query(Attendance).filter(Attendance.id == attendance_id).first()

    if row is None and person_id is not None:
        row = (
            db.query(Attendance)
            .filter(Attendance.person_id == person_id, Attendance.day == selected_day)
            .first()
        )

    if row is None:
        # No existing attendance row: allow creating when marking.
        if status == "unmarked":
            db.close()
            return RedirectResponse(url=redirect_url, status_code=303)

        if person_id is None:
            db.close()
            return {"error": "person_id required"}

        allowed_ids = _allowed_person_ids(db, request, role, section_id)
        if allowed_ids is not None and person_id not in allowed_ids:
            db.close()
            return RedirectResponse(url=redirect_url, status_code=303)

        person = db.query(Person).filter(Person.id == person_id).first()
        if not person:
            db.close()
            return {"error": "Student not found"}
        if person.is_blocked:
            db.close()
            return RedirectResponse(url=redirect_url, status_code=303)

        row = Attendance(
            person_id=person.id,
            roll_no=person.roll_no,
            name=person.name,
            day=selected_day,
            status="marked",
            marked_at=datetime.utcnow(),
        )
        db.add(row)
        db.commit()
        db.close()
        return RedirectResponse(url=redirect_url, status_code=303)

    # Update existing attendance row
    allowed_ids = _allowed_person_ids(db, request, role, section_id)
    if allowed_ids is not None and row.person_id not in allowed_ids:
        db.close()
        return RedirectResponse(url=redirect_url, status_code=303)

    person = db.query(Person).filter(Person.id == row.person_id).first()
    if person and person.is_blocked:
        db.close()
        return RedirectResponse(url=redirect_url, status_code=303)
    previous_status = (row.status or "unmarked")
    row.status = status
    if status == "marked" and previous_status != "marked":
        row.marked_at = datetime.utcnow()
        # keep name/roll synced with Person
        if person:
            row.roll_no = person.roll_no
            row.name = person.name
    db.commit()
    db.close()
    return RedirectResponse(url=redirect_url, status_code=303)