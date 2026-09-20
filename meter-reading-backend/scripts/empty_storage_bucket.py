"""
scripts/empty_storage_bucket.py

Deletes EVERY file in the Supabase Storage bucket (SUPABASE_STORAGE_BUCKET).
Use before a DB reset if you also want to reclaim storage space, since
dropping the `gauges` table does NOT delete the actual photo files --
they live in Storage, a completely separate system from Postgres.

Run from the backend root, with .env already set (SUPABASE_URL,
SUPABASE_SERVICE_KEY, SUPABASE_STORAGE_BUCKET):

    python3 scripts/empty_storage_bucket.py

DESTRUCTIVE -- there's no undo. Double-check SUPABASE_STORAGE_BUCKET is
the bucket you actually mean before running this.
"""

import os

import httpx
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
SUPABASE_BUCKET = os.environ.get("SUPABASE_STORAGE_BUCKET", "gauge-photos")

HEADERS = {"Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}


def list_all_files() -> list[str]:
    response = httpx.post(
        f"{SUPABASE_URL}/storage/v1/object/list/{SUPABASE_BUCKET}",
        headers=HEADERS,
        json={"prefix": "", "limit": 1000},
        timeout=15.0,
    )
    response.raise_for_status()
    return [entry["name"] for entry in response.json()]


def delete_files(paths: list[str]) -> None:
    response = httpx.request(
        "DELETE",
        f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}",
        headers=HEADERS,
        json={"prefixes": paths},
        timeout=15.0,
    )
    response.raise_for_status()


if __name__ == "__main__":
    paths = list_all_files()

    if not paths:
        print(f"'{SUPABASE_BUCKET}' 버킷이 이미 비어있어요.")
    else:
        print(f"'{SUPABASE_BUCKET}' 버킷에서 {len(paths)}개 파일 삭제 중...")
        delete_files(paths)
        print("삭제 완료.")