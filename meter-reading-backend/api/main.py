"""
api/main.py

FastAPI application entry point. Run locally with:
    uvicorn api.main:app --reload

Kept deliberately thin -- this file's only job is to create the app and
wire in routers. Actual endpoint logic lives in api/routes/*.py, same
reasoning as keeping workers/*.py thin wrappers around the pipeline
modules rather than inlining logic into the Celery task functions.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import gauges, jobs

app = FastAPI(title="Analog Meter Reading API")

# The frontend (Vite dev server, localhost:5173) and the API (localhost:8000)
# are different origins as far as the browser is concerned -- without this,
# the browser blocks the request before it ever reaches this server, which
# is exactly the silent "Upload failed" behavior seen from the frontend.
# localhost:5173 stays for local dev; the production frontend domain
# will be app.analog-gauge-reader.website (Vercel, custom domain) --
# added now even before that deploy exists, since the domain itself
# (unlike Vercel's auto-generated *.vercel.app URL) won't change later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://app.analog-gauge-reader.website",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(gauges.router)
app.include_router(jobs.router)


@app.get("/health")
def health() -> dict:
    """Simple liveness check -- not part of the pipeline, just confirms the API process is up."""
    return {"status": "ok"}