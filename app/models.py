from sqlalchemy import Column, Integer, String, LargeBinary, DateTime, UniqueConstraint, Index, Boolean
from app.database import Base
from datetime import datetime


class UserAccount(Base):
    __tablename__ = "user_accounts"

    id = Column(Integer, primary_key=True, index=True)
    role = Column(String, index=True, nullable=False)  # admin | faculty | trainer
    username = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=True)
    password_hash = Column(String, nullable=False)
    photo_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class Person(Base):
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, index=True)
    # Treat `id` as Registration No (Regn No) for simplicity.
    roll_no = Column(String, unique=True, index=True, nullable=True)
    name = Column(String, unique=True)
    embedding = Column(LargeBinary)  # Serialized numpy array
    is_blocked = Column(Boolean, default=False, nullable=False)
    blocked_reason = Column(String, nullable=True)


class Section(Base):
    __tablename__ = "sections"

    id = Column(Integer, primary_key=True, index=True)
    course_code = Column(String, index=True, nullable=False)
    course_name = Column(String, nullable=False)
    faculty_uid = Column(String, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class SectionStudent(Base):
    __tablename__ = "section_students"

    id = Column(Integer, primary_key=True, index=True)
    section_id = Column(Integer, index=True, nullable=False)
    person_id = Column(Integer, index=True, nullable=False)

    __table_args__ = (
        UniqueConstraint("section_id", "person_id", name="uq_section_student"),
        Index("ix_section_student_section_person", "section_id", "person_id"),
    )


class Attendance(Base):
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, index=True, nullable=False)
    roll_no = Column(String, index=True, nullable=True)
    name = Column(String, index=True, nullable=False)
    day = Column(String, index=True, nullable=False)  # YYYY-MM-DD (local)
    marked_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    status = Column(String, index=True, nullable=False, default="marked")  # marked | unmarked

    __table_args__ = (
        UniqueConstraint("person_id", "day", name="uq_attendance_person_day"),
        Index("ix_attendance_day_marked_at", "day", "marked_at"),
    )
