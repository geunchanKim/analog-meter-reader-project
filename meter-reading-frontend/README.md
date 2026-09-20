# Analog Meter Reading — Frontend

React (Vite + TypeScript) client for uploading a gauge photo, reviewing
the detected keypoints, choosing/swapping/correcting the OCR'd min and
max values, previewing then judging the computed reading, and browsing
past readings. Live at `https://app.analog-gauge-reader.website`.

## How this got built, and why it looks the way it does

The plan going in was to port an existing HTML/CSS mockup. What actually
shipped diverged from it in a few deliberate ways, each traceable to a
specific problem:

**No manual min/max text-entry fields as the primary flow.** The
mockup's blank number inputs were replaced with the backend's actual
shape: top-3 OCR candidates per position (`pos_a_candidates` /
`pos_b_candidates`), radio-selected, with a swap control for when the
model's min/max guess is backwards. A manual override input ("Other...")
was added to each column afterward, once a real test photo came back
with its true max value ("60") missing from all three OCR candidates --
without an escape hatch, that photo (and any like it) would simply have
been unusable.

**The upload control is a viewfinder (corner brackets), not a circle.**
An earlier version was shaped like a circular gauge dial, echoing the
subject matter -- but that implied the app only reads round gauges, when
Stage 1 on the backend detects both circular and rectangular gauge
housings. The shape was changed to something that doesn't silently claim
a limitation that isn't real.

**Calculate → preview → judge, not pick-and-submit.** The first version
of the candidate screen let a person pick candidates and immediately hit
Correct/Incorrect/Unclear. That doesn't actually work: judging a reading
"incorrect" requires having seen what the calculated value *is* first,
and the old flow only revealed the computed value *after* a judgment was
already submitted. `POST /gauges/{id}/preview` (computes a reading
without saving) was added specifically to fix that ordering -- a person
now sees the number, then judges it. Choosing "Incorrect" reveals an
actual-value input, so the real reading (as read by eye) gets recorded
alongside the wrong computed one, for later error analysis.

**Design tokens (IBM Plex Sans/Mono, graphite + needle-tip-red accent)**
were pulled from the gauges' own material language -- steel bezels,
bone-white dial faces, red needle tips -- rather than a generic dark
theme, deliberately avoiding the near-black-plus-one-bright-accent
default look.

**The history sidebar opens a centered modal, not an inline expansion.**
The first version expanded a row's detail directly inside the 260px
sidebar column, which made the photo and keypoints hard to actually see.
Clicking a row now opens a centered modal instead, with the full photo,
keypoint overlay, and a Min/Max/Actual/Judgment breakdown.

## Requirements

- Node.js 18+
- The backend API reachable (local `http://localhost:8000`, or the
  deployed `https://api.analog-gauge-reader.website`)

## Repo layout

```
frontend/
├── src/
│   ├── api/
│   │   └── client.ts                   # thin wrapper over every backend endpoint --
│   │                                    #   components never construct a fetch() call directly
│   ├── hooks/
│   │   └── useJobPolling.ts            # polls GET /jobs/{job_id} until a terminal status
│   ├── components/
│   │   ├── PhotoDropzone.tsx/.css      # upload control (viewfinder-style)
│   │   ├── ImageKeypointOverlay.tsx/.css  # photo + SVG keypoint overlay, optional crop-to-bbox
│   │   └── HistorySidebar.tsx/.css     # persistent left sidebar; click a row -> centered
│   │                                    #   modal with photo/keypoints/stats
│   ├── screens/
│   │   ├── UploadScreen.tsx/.css       # upload + poll; photo preview persists through processing
│   │   └── CandidateSelectionScreen.tsx/.css
│   │                                    #   candidates + swap + manual override + Ask AI +
│   │                                    #   calculate/preview/judge + color-coded result
│   ├── styles/
│   │   └── tokens.css                  # design tokens (color/font CSS variables)
│   ├── App.tsx / App.css               # screen state machine + sidebar layout shell
│   └── main.tsx
├── .env.example
└── package.json
```

## Environment variables

```
VITE_API_BASE_URL=https://api.analog-gauge-reader.website
```

Use `http://localhost:8000` for local backend dev. This must be set in
Vercel's Project Settings → Environment Variables for the deployed
build -- the local `.env` file isn't read by Vercel.

## Local setup

```bash
npm install
cp .env.example .env   # set VITE_API_BASE_URL
npm run dev
```

Linting is Oxlint (`.oxlintrc.json`), chosen over ESLint for near-zero
config and speed, given the project's scale and pace.

## Backend contract this UI is built against

- `POST /gauges` — multipart upload, returns `job_id`
- `GET /jobs/{job_id}` — poll until `status` is a terminal value (`done`, `fail`, `retake`); response includes `gauge_id` once available
- `GET /gauges/{gauge_id}` — `image_url`, `bbox`, `keypoints`, `pos_a_candidates`/`pos_b_candidates` (each entry has `text`, `parsed_value`, `rank`, `is_forced_zero`)
- `POST /gauges/{gauge_id}/preview` — `{chosen_min_value, chosen_max_value, swapped}` → `{computed_value}`, no persistence
- `POST /gauges/{gauge_id}/selection` — adds `judgment` and optional `actual_value` to the preview's inputs, persists, returns the final `computed_value`
- `GET /gauges` — history list, most recent first, each entry includes `image_url`
- `POST /gauges/{gauge_id}/explain` — `{position_a, position_b}` advisory text from the AI agent (see backend README for its exact scope)

## Deployment

**Vercel**, deployed from this GitHub repo (auto-detected Vite build:
`vite build` → `dist`). Custom domain `app.analog-gauge-reader.website`
is a CNAME (DNS-only, not proxied) added in Cloudflare, pointing at the
Vercel-provided target -- proxying through Cloudflare as well would have
conflicted with Vercel's own certificate issuance.

The backend it talks to is not on a conventional host -- see the backend
README's "How this got built" section for why (Cloudflare Tunnel from a
personal laptop, not a rented VPS).

## Testing

No committed test suite. Every screen and flow (upload → poll, candidate
selection + swap + manual override, calculate → preview → judge, Ask AI,
history sidebar + modal) was verified interactively during development,
but those checks weren't kept as files, on the same judgment as the
backend: not warranted at this project's current stage.