"""
db/storage.py

Uploads the original uploaded photo to Supabase Storage and returns its
public URL. This is what makes Gauge.image_url (a column that's existed
in the schema from the start, unused until now) actually get filled in --
without it, no server-side copy of the photo existed anywhere; only a
client-side object URL that dies with the browser tab/session.
"""

import logging
import os
import uuid

import httpx

logger = logging.getLogger(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET = os.environ.get("SUPABASE_STORAGE_BUCKET", "gauge-photos")


def upload_gauge_photo(image_bytes: bytes, gauge_id: uuid.UUID | str) -> str:
    """
    Uploads the photo under a path keyed by gauge_id, and returns its
    public URL (this assumes the bucket is configured public -- for a
    private bucket, this would need to return/generate a signed URL
    instead, which expires and needs re-issuing on each fetch).

    Raises on misconfiguration or a failed upload -- callers (torch_worker.py)
    decide whether that should just be logged and skipped, since the
    actual gauge reading doesn't depend on this succeeding; only the
    history sidebar's photo display does.
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_KEY not configured")

    path = f"{gauge_id}.jpg"
    upload_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{path}"

    response = httpx.post(
        upload_url,
        headers={
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": "image/jpeg",
            "x-upsert": "true",  # overwrite if this gauge_id was already uploaded (retry-safe)
        },
        content=image_bytes,
        timeout=15.0,
    )
    response.raise_for_status()

    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{path}"