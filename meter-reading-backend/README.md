# Analog Meter Reading — Backend

FastAPI + Celery pipeline that reads an analog gauge's value from a
photo: detect the gauge, find its keypoints, OCR the min/max scale
labels, compute the needle's value from the angles, persist everything,
and let a human confirm or correct the result. Live at
`https://api.analog-gauge-reader.website`.

## How this got built, and why it looks the way it does

The pipeline was built stage by stage, and each stage's shape came from
a real problem encountered while building the one before it, not from
an upfront design:

**Stage 1 (gauge detection) and Stage 2 (keypoints)** started as
separate exploration notebooks -- Stage 1 replaced a failing
HoughCircles-only approach with an overlap-based candidate merging
algorithm; Stage 2 went through three YOLOv8-pose training iterations to
fix train/test leakage (Roboflow's augmentation copies were duplicating
images across the split, inflating mAP) and a systematic min/max
keypoint swap (caused by horizontal-flip augmentation without adjusting
`flip_idx`).

**Stage 3 (OCR) exists as its own module family** (`preprocessing.py`,
`ocr_runner.py`, `candidate_scoring.py`, `search_strategy.py`,
`pipeline_stage3.py`) because a single PaddleOCR call on a raw crop
wasn't reliable enough -- the top-3-candidates-per-position design (not
a single "best guess") came directly out of watching real OCR failures:
often the correct reading was in PaddleOCR's second or third candidate,
not its first. That's also why `Candidate.is_forced_zero` exists: OCR
regularly failed to read a printed "0" at all, so a synthetic "0"
candidate is inserted for the min-labeled position as a fallback,
flagged so nothing downstream (a human, or the AI agent) mistakes it for
an actual OCR read.

**Why Stage 1+2 and Stage 3 are separate Celery workers, not one
process:** confirmed directly that PyTorch (ultralytics) and PaddlePaddle
(paddleocr) crash if imported into the same process --

```
RuntimeError: (PreconditionNotMet) Tensor holds no memory. Call Tensor::mutable_data firstly.
```

`workers/torch_worker.py` and `workers/paddle_worker.py` run as
independent processes, handing off by Celery task name
(`celery_app.send_task(...)`) rather than importing each other. This
same rule is why the API server never imports either worker module
directly either -- `POST /gauges` enqueues Stage 1 by name, so the API
process never loads a YOLO model just to reference a task.

**Why min/max/swap is never auto-decided:** early on, an attempt to
auto-detect which of two keypoints was "min" vs "max" (and auto-correct
if the model's guess looked backwards) kept getting it wrong in ways
that were hard to detect programmatically. The system now always
surfaces the model's top-3 OCR candidates for a human to pick from, with
a manual swap control -- the pipeline's job is narrowed to "give a human
good candidates," not "always be right." That same philosophy carries
into the reading-confirmation flow: `POST /gauges/{id}/preview` computes
a value without saving it, specifically so a human sees the number
*before* judging it correct/incorrect -- judging something you haven't
seen yet isn't a real check.

**Why FastAPI + SQLAlchemy + Postgres (Supabase):** FastAPI for the
async-friendly, typed-request/response ergonomics with Celery already in
the stack; SQLAlchemy so `db/crud.py` holds every save/query as named
functions instead of raw SQL scattered through the workers and routes;
Supabase specifically for a free, hosted Postgres instance (session
pooler, not transaction pooler -- this app's clients are a couple of
long-lived worker processes, not many short-lived serverless ones, so
session mode's fewer restrictions suit it better) plus its Storage
product for the photo files themselves (`db/storage.py` uploads the
original photo on Stage 1 success; `Gauge.image_url` holds the public
URL; a failed upload doesn't block the reading -- it's best-effort,
logged, not raised).

**Why Groq for the AI agent, and what it actually does:** the agent's
only current job is `POST /gauges/{gauge_id}/explain` -- it looks at the
same top-3 OCR candidates a human sees and writes one short, hedged
sentence per position (e.g. "0 looks most plausible as a min value")
commenting on plausibility. It never picks a value, never touches the
DB, and has no memory across calls -- it's a second opinion offered
alongside the candidates, not a decision-maker. Groq was chosen for its
free tier and OpenAI-compatible API (so `agent/llm_client.py` uses the
`openai` SDK, just pointed at Groq's base URL -- switching providers
later is a one-line change, not a rewrite). The system prompt
(`agent/llm_client.py`) exists specifically to keep the model from
drifting into overconfidence -- an early version of the prompt let it
praise whichever candidate ranked first regardless of whether the
numbers actually made sense together; the current version explicitly
tells it to say so when candidates look inconsistent, and to never
claim a forced-zero candidate was "confidently detected."

**Why the deployment isn't Docker on a rented VPS, despite that being
the original plan:** two real blockers, discovered in this order --
PaddlePaddle has no ARM build (`pip install paddlepaddle` has no
aarch64 wheel), which ruled out the free ARM-based VPS tier that had
enough RAM headroom; and separately, signing up for a payment method on
the x86 VPS options failed while traveling abroad without a way to
receive Korean SMS verification. What's running instead is the same
5-process local setup (below) on a personal laptop, exposed via
**Cloudflare Tunnel** -- which needs no credit card or phone
verification, only a domain pointed at Cloudflare's nameservers. The
trade-off: this is only reachable while the laptop is on and the tunnel
process is running, not a conventional always-on server.

## Requirements

- Python 3.11+ (developed against 3.13)
- A YOLOv8-pose checkpoint (Stage 2 keypoint model)
- Redis (Celery broker)
- A Postgres database (Supabase, in practice)
- A Supabase Storage bucket (for saving uploaded photos)
- A Groq API key (free tier)

## Repo layout

```
backend/
├── pointer_math/
│   └── angle_utils.py       # angle math + final value calc; resolve_reading_fraction()
│                             #   picks major- vs minor-arc scale sweep based on where
│                             #   the tip actually falls (fixes a real bug found on a
│                             #   panel ammeter whose scale only sweeps ~94deg -- see
│                             #   the function's docstring)
├── stage1_2/
│   ├── gauge_detection.py   # Stage 1: locate + crop the gauge (circle + rectangle)
│   ├── key_points.py        # Stage 2: YOLO keypoint model wrapper
│   └── qc_gate.py           # confidence + arc-span checks before Stage 3
├── stage3/
│   ├── preprocessing.py     # crop / upscale / binarize / rotate helpers
│   ├── ocr_runner.py        # PaddleOCR wrapper + number/label filtering
│   ├── candidate_scoring.py # collect + score OCR candidates
│   ├── search_strategy.py   # 4-stage progressive search
│   └── pipeline_stage3.py   # assembles the above into one Stage 3 call
├── db/
│   ├── models.py            # SQLAlchemy models: Job, Gauge, Keypoints, Candidate, Selection
│   ├── session.py           # engine/session setup, init_db(), loads .env
│   ├── crud.py              # every save/query the workers + API call
│   └── storage.py           # uploads photos to Supabase Storage, returns public URL
├── agent/
│   └── llm_client.py        # CandidateExplainer -- Groq call, advisory only
├── api/
│   ├── main.py               # FastAPI app + CORS
│   └── routes/
│       ├── gauges.py         # POST /gauges, GET /gauges/{id}, .../preview, .../selection,
│       │                      #   GET /gauges (history), .../explain
│       └── jobs.py           # GET /jobs/{job_id}
├── workers/
│   ├── celery_app.py         # shared Celery config, task_routes (torch_queue / paddle_queue)
│   ├── torch_worker.py       # Stage1+2+QC, DB save, photo upload, hands off to paddle_worker
│   └── paddle_worker.py      # Stage3, DB save, JSON-safety conversion for the Celery result
├── analysis/
│   └── precompute_keypoints.py  # offline batch script for validating qc_gate thresholds
└── requirements.txt
```

## Environment variables

```
DATABASE_URL=postgresql+psycopg://postgres.<project-ref>:<password>@<host>:5432/postgres
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_SERVICE_KEY=<service_role key>
SUPABASE_STORAGE_BUCKET=gauge-photos
GROQ_API_KEY=<groq key>
GROQ_MODEL=llama-3.3-70b-versatile
KEYPOINT_MODEL_PATH=models/<name>/best.pt
OCR_DEVICE=cpu
```

The `+psycopg` in `DATABASE_URL` matters -- without it, SQLAlchemy
defaults to looking for `psycopg2` (not installed) instead of `psycopg`
v3 (what's actually in `requirements.txt`).

## Local setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt   # add --break-system-packages if your pip complains
```

Set `.env`, then create the DB tables:
```bash
python3 -c "from db.session import init_db; init_db()"
```

## Running the whole stack locally

Five processes, five terminals:

```bash
# 1. Redis
redis-server

# 2. Stage 1+2 worker
celery -A workers.celery_app worker -Q torch_queue --loglevel=info -n torch@%h -I workers.torch_worker

# 3. Stage 3 worker
celery -A workers.celery_app worker -Q paddle_queue --loglevel=info -n paddle@%h -I workers.paddle_worker

# 4. API server
uvicorn api.main:app --reload

# 5. (only when exposing this publicly) Cloudflare Tunnel
cloudflared tunnel run meter-reading
```

Verify: `curl http://localhost:8000/health` → `{"status": "ok"}`.

## Deployment

Live at `https://api.analog-gauge-reader.website` via Cloudflare Tunnel
(see "How this got built" above for why, not a rented VPS):

```bash
brew install cloudflared
cloudflared tunnel login
cloudflared tunnel create meter-reading
# ~/.cloudflared/config.yml:
#   tunnel: <tunnel-id>
#   credentials-file: /Users/<you>/.cloudflared/<tunnel-id>.json
#   ingress:
#     - hostname: api.analog-gauge-reader.website
#       service: http://localhost:8000
#     - service: http_status:404
cloudflared tunnel route dns meter-reading api.analog-gauge-reader.website
cloudflared tunnel run meter-reading
```

CORS (`api/main.py`) allows `http://localhost:5173` (local frontend dev)
and `https://app.analog-gauge-reader.website` (deployed frontend).

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/gauges` | Upload a photo (multipart), enqueue Stage 1 |
| `GET` | `/jobs/{job_id}` | Poll job status; includes `gauge_id` once a Gauge row exists |
| `GET` | `/gauges/{gauge_id}` | `image_url`, `bbox`, `keypoints`, `pos_a_candidates`/`pos_b_candidates` (each with `is_forced_zero`) |
| `POST` | `/gauges/{gauge_id}/preview` | Computes a reading from a chosen min/max WITHOUT saving |
| `POST` | `/gauges/{gauge_id}/selection` | Saves the human's final min/max + swap + judgment (+ `actual_value` if incorrect) |
| `GET` | `/gauges` | History, most recent first, with `image_url` per entry |
| `POST` | `/gauges/{gauge_id}/explain` | AI agent's plausibility comments -- see "Why Groq" above for its exact scope |
| `GET` | `/health` | Liveness check |

## Database

Tables: `jobs`, `gauges`, `keypoints`, `candidates`, `selections` --
schema lives in `db/models.py`, created via `db.session.init_db()`.

## Testing

No committed test suite. Every module was verified interactively during
development against real and synthetic data (thresholds in `qc_gate.py`,
the arc-selection fix in `angle_utils.py`, each DB save/query function,
every API endpoint, the Celery handoff) -- those checks weren't kept as
files, on the judgment that a formal suite isn't warranted at this
project's current stage.