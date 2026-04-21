**🤖 FaceApp – AI Face Recognition**
*🚀 Overview*

FaceApp is an advanced AI-powered face recognition system capable of recognizing multiple faces in bulk with up to 99% accuracy.
It uses RetinaFace for precise face detection and ArcFace (InsightFace) for accurate face embedding and identity recognition.

The backend is powered by FastAPI, making the entire system asynchronous, lightweight, and ultra-fast — suitable for real-world deployments such as classrooms, organizations, and event-based attendance systems.

*✨ Key Features*

🧠 99% Accuracy using ArcFace embeddings from InsightFace.

👥 Bulk Recognition: Recognizes multiple faces in a single image efficiently.

⚡ Asynchronous Backend: Built with FastAPI for high performance and scalability.

🖼️ Proof of Recognition: Supports storing and visualizing recognized group images.

🔒 Modular & Secure: Can easily integrate into larger systems like attendance tracking, security monitoring, or analytics dashboards.

*🏗️ Tech Stack*
Component	Technology Used
Backend Framework	FastAPI
Face Detection	RetinaFace
Face Recognition	ArcFace (InsightFace)
Programming Language	Python
Server/Deployment	Uvicorn / Docker (optional)
*🧬 System Workflow*
[Input Image(s)]
       ↓
[RetinaFace Detection]
       ↓
[ArcFace Embeddings Extraction]
       ↓
[Identity Matching]
       ↓
[Output → Recognized Group Image(s)]



📷 Example Results:

Classroom group recognition

Office meeting face identification

Event crowd recognition

![Group Recognition Example 1](recognized_c384ca0e547e4697a2a977f6b21afcf4.jpg)
![Group Recognition Example 2](recognized_6c317938e85444bfb586f43c1e552ce4.jpg)




*⚙️ Installation & Run Locally*
# Clone the repository
git clone https://github.com/jpmandal-02/Multiface-Recognition-Fastapi.git
cd Multiface-Recognition-Fastapi

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn app.main:app --reload


Once started, open:

http://127.0.0.1:8000/docs


to explore the interactive API documentation and test the recognition endpoints.

🌐 Future Scope

🚀 Upcoming API Release:
A public FaceApp API will soon be launched — enabling developers and organizations to automate attendance systems across schools, universities, workplaces, and event spaces with a few API calls.

Other planned upgrades:

🎥 Real-time video feed support

☁️ Cloud-based storage and recognition

📊 Analytics dashboard for attendance tracking

🔔 Automatic alert or notification integration

🤝 Contributing

Pull requests and feature suggestions are always welcome!
If you find this project useful, consider ⭐ starring the repo and sharing it with others.

## Docker Deployment with Prometheus + Grafana

This repo now includes a Docker-based monitoring stack:

- FastAPI app on port `8010`
- Prometheus on port `9100`
- Grafana on port `3100`
- cAdvisor on port `18080` (container CPU/memory metrics)

Before running, make sure Docker Desktop (daemon) is started.

### 1) Start the app only

```bash
docker compose up -d --build
```

### 2) Start the monitoring stack too

```bash
docker compose --profile monitoring up -d --build
```

### 3) Open services

- App: `http://localhost:8010/login`
- App metrics: `http://localhost:8010/metrics`
- Prometheus: `http://localhost:9100`
- Grafana: `http://localhost:3100`

login:

- Username: `1234`
- Password: `5678`

### 4) Add Prometheus as Grafana data source

In Grafana:

1. Go to **Connections** -> **Data sources**.
2. Add **Prometheus**.
3. Set URL to `http://prometheus:9090`.
4. Click **Save & test**.

### 5) Useful PromQL queries for dashboards

Request rate:

```promql
sum(rate(http_requests_total[1m]))
```

P95 latency:

```promql
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))
```

5xx error rate:

```promql
sum(rate(http_requests_total{status=~"5.."}[5m]))
```

Container CPU (app):

```promql
sum(rate(container_cpu_usage_seconds_total{name=~".*ams_app.*"}[1m]))
```

Container memory (app):

```promql
sum(container_memory_usage_bytes{name=~".*ams_app.*"})
```

### 6) Stop stack

```bash
docker compose down
```

### Performance Notes (Docker)

- The first app startup can be slow because InsightFace models are downloaded once.
- Model cache is persisted in Docker volume `insightface_cache`, so next restarts are faster.
- Live camera frames are compressed/downscaled for lower inference latency.

If you need faster response on CPU-only machines, keep:

- `FACEAPP_DET_SIZE=512` (or reduce to `448`)
- lower camera frame upload width / interval (already tuned in `live.html`)

### Data Persistence (Important)

Student data and attendance are stored in SQLite. In Docker, persistence is now configured with named volumes:

- `app_data` -> `/app/data` (database file)
- `app_uploads` -> `/app/static/uploads` (uploaded photos)

Use normal stop/start commands to keep data:

```bash
docker compose down
docker compose up -d
```

Do **not** use `-v` unless you intentionally want to erase data:

```bash
docker compose down -v
```

## CPU Alert Email Notifications (Prometheus + Alertmanager)

This setup sends an email when AMS app CPU usage stays above 50 percent for 2 minutes.

### 1) Configure Gmail SMTP credentials

Copy `.env.example` to `.env` and update the SMTP values:

```bash
cp .env.example .env
```

Set:

- `ALERT_SMTP_FROM`
- `ALERT_SMTP_USERNAME`
- `ALERT_SMTP_PASSWORD` (Gmail App Password)

### 2) Start/restart stack

```bash
docker compose down
docker compose up -d --build
```

### 3) Verify alert components

- Prometheus: `http://localhost:9100`
- Alertmanager: `http://localhost:9300`

In Prometheus:

1. Go to **Status** -> **Targets** and verify all targets are up.
2. Go to **Alerts** and check alert `AppHighCpuUsage`.

### 4) Alert rule details

Rule file: `monitoring/alert-rules.yml`

- Alert: `AppHighCpuUsage`
- Condition: app CPU > 50 percent
- Duration: 2 minutes
- Receiver email: `satyamkumarsingh705071@gmail.com`

### 5) Quick test by generating CPU load

```bash
docker exec -it ams_app sh -c "python - <<'PY'
import time
x = 0
for _ in range(40_000_000):
       x += 1
time.sleep(150)
print(x)
PY"
```

After about 2 minutes, you should receive an alert email.

## Railway MySQL Setup

This project can run with SQLite locally or MySQL on Railway.

### 1) Create a MySQL database in Railway

In Railway:

1. Create a new project.
2. Add the `MySQL` service.
3. Open the MySQL service and copy its connection string.

### 2) Set `DATABASE_URL`

Add this environment variable in Railway for your app service:

```bash
DATABASE_URL=mysql+pymysql://USER:PASSWORD@HOST:PORT/DATABASE
```

If Railway gives you a URL starting with `mysql://`, this app now converts it automatically.

### 3) Install dependency

Make sure the app installs:

```bash
pip install -r requirement.txt
```

This now includes `pymysql`, which SQLAlchemy uses to connect to Railway MySQL.

### 4) Deploy

On Railway, deploy the app service with the same project code and set all required environment variables such as:

- `DATABASE_URL`
- `FACEAPP_ADMIN_ID`
- `FACEAPP_ADMIN_PASSWORD`
- `FACEAPP_SESSION_SECRET`

### 5) Local development

If `DATABASE_URL` is not set, the app continues using local SQLite:

```bash
sqlite:///./embeddings.db
```

## Railway Port Fix

Railway only exposes the app when the running server listens on Railway's assigned `PORT`.

This project now starts Uvicorn with:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

That means:

- `0.0.0.0` allows Railway to reach the app from outside the container
- `${PORT:-8000}` uses Railway's dynamic port in production and `8000` locally

If Railway still shows `Unexposed service`, open your Railway service settings and make sure the web service is using the app container from this repository, then redeploy.

## Vercel + Railway

This repo is a server-rendered FastAPI app, not a separate frontend/backend monorepo.

- Deploy the real app on Railway
- If you want the public URL on Vercel, use Vercel as a rewrite/proxy in front of Railway

See [DEPLOYMENT.md](./DEPLOYMENT.md) for the exact setup and `vercel.json.example`.
