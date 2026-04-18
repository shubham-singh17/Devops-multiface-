# Deployment Guide

## Important

This repository is not split into a separate frontend and backend app.
The HTML pages in `templates/` are rendered directly by the FastAPI server in `app/main.py`.

That means:

- Railway is the correct place to run the real application
- Vercel cannot host this "frontend" as an independent app unless you rebuild the UI into React/Next/static pages
- If you still want the public website on Vercel, the practical option is to use Vercel as a reverse proxy in front of Railway

## Recommended Setup

Use:

- Railway for the FastAPI app and database
- Vercel only as the public domain/proxy layer

Browser flow:

`User -> Vercel domain -> rewrite/proxy -> Railway app`

Because every route is proxied to Railway, the existing forms, cookies, and relative URLs continue to work.

## 1. Deploy The App To Railway

### Create the services

In Railway:

1. Create a new project
2. Add a `MySQL` service
3. Add a service from this GitHub repository for the app

### Railway environment variables

Set these variables in the app service:

- `DATABASE_URL`
- `FACEAPP_ADMIN_ID`
- `FACEAPP_ADMIN_PASSWORD`
- `FACEAPP_SESSION_SECRET`
- `FACEAPP_PROVIDERS=CPUExecutionProvider`
- `FACEAPP_DET_SIZE=512`
- `OMP_NUM_THREADS=2`
- `OPENBLAS_NUM_THREADS=2`

Optional mail variables for alerts:

- `ALERT_SMTP_SMARTHOST`
- `ALERT_SMTP_FROM`
- `ALERT_SMTP_USERNAME`
- `ALERT_SMTP_PASSWORD`
- `ALERT_EMAIL_TO`

### Notes

- This repo already includes a `Dockerfile`, so Railway can deploy it directly
- The container already starts on `0.0.0.0` and uses `${PORT}`
- The first start can be slow because InsightFace downloads models

### Persistent storage

If you want uploads and SQLite-style local files to persist on Railway, attach a volume and mount it to:

`/app/data`

If you use Railway MySQL for `DATABASE_URL`, database records will persist there, but uploaded profile photos under `static/uploads` still need persistent storage if you do not want them lost on redeploy.

Recommended extra volume mount:

`/app/static/uploads`

## 2. Verify Railway First

After Railway deploys, open:

- `https://your-railway-app.up.railway.app/login`

Make sure:

- the login page loads
- static assets load
- admin login works
- uploads and training endpoints work

Do this before adding Vercel.

## 3. Put Vercel In Front

### What Vercel should do

Vercel should proxy all routes to Railway.
Do not try to deploy this Python app itself to Vercel serverless functions because the face-recognition stack is too heavy for that model.

### File to use

This repo includes `vercel.json.example`.

Create a real `vercel.json` from it and replace:

`https://your-railway-app.up.railway.app`

with your actual Railway public domain.

Example:

```json
{
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "https://my-faceapp-production.up.railway.app/$1"
    }
  ]
}
```

### Deploy on Vercel

1. Import this repository into Vercel
2. Keep the project as a simple static project
3. Add the final `vercel.json`
4. Deploy

Now requests to the Vercel URL will be forwarded to Railway.

## 4. Domain Setup

If you want your custom domain on Vercel:

1. Attach the domain in Vercel
2. Keep the rewrite pointing to Railway

If you want your custom domain directly on Railway instead, you can skip Vercel completely.

## 5. Best Practical Recommendation

For this specific codebase, the simplest and most reliable production setup is:

- Deploy the full app on Railway
- Use Railway's public URL directly

Use Vercel only if you specifically want:

- Vercel-managed domain/DNS
- Vercel preview URLs
- a proxy layer in front of Railway

## 6. If You Truly Want A Separate Frontend

That would require a real refactor:

- build a separate frontend app, such as Next.js
- move UI actions from form posts to API calls
- add CORS/cookie strategy
- expose the FastAPI app as a pure backend API

This repo is not structured that way today.
