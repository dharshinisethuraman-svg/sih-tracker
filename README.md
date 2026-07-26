# Skill Training Centre — Enrolment & Placement Tracker

**SIH 2026 Internal Practical Assessment** ·

## Problem, in two lines

A rural skill training centre enrols candidates, trains them, and is
supposed to report how many find work — but enrolment forms, attendance
registers, and phone-call placement follow-ups live in three separate
places, so the centre can't tell which courses actually lead to
employment. This app puts all three in one place and shows the
completion and placement rate per course.

## What's in this repo (maps to the assessment tasks)

| Task | Where |
|---|---|
| 1. Sample data | `data/generate_sample_data.py` → `data/sample_data.csv` |
| 2. Register (form + backend + validation) | `app.py`, `db.py`, `templates/` |
| 3. Image classifier | `ml/generate_images.py`, `ml/train_classifier.py` |
| 4. Containerise | `Dockerfile`, `docker-compose.yml`, `.env.example` |
| 5. Integration | certificate check wired into the placement-update screen |
| 6. This README + screenshots + demo video |

## A note on Task 3 (read this first)

The brief asks for an image classifier but doesn't say what it should
classify for *this* problem statement (that part of the brief looks
templated across several different problem statements). Since it wasn't
specified, this project uses a **certificate authenticity check**:
given a photo of a candidate's completion certificate, predict whether
it looks **genuine** or **tampered**. It's a natural fit for a skill
centre — coordinators are handed printed certificates with no reliable
way to spot an edited name or date. The images are generated
synthetically (`ml/generate_images.py`), which the brief explicitly
allows ("you may photograph or generate them").

## How the two placement-rate numbers are defined

- **Completion rate** = completed candidates ÷ all enrolled candidates, per course.
- **Placement rate** = placed candidates ÷ **completed** candidates, per course
  (of the people who finished, what % got a job). Candidates who never
  completed the course are excluded from this rate — including them
  would unfairly penalise a course for candidates who dropped out
  before finishing.
- `placement_status = "Unknown"` (follow-up call never made) is **not**
  counted as "Not Placed" anywhere — treating unknowns as failures
  would understate every course's real performance.

## Field meanings (`candidate_id, name, course, batch, enrolment_date, attendance_pct, completed, placement_status, employer`)

| Field | Type | Meaning |
|---|---|---|
| `candidate_id` | int | Primary key, auto-generated |
| `name` | text | Candidate's full name |
| `course` | text | One of the 5 courses the centre runs |
| `batch` | text | Batch code, e.g. `2026-A` |
| `enrolment_date` | date | When the candidate joined |
| `attendance_pct` | float 0–100, nullable | % of sessions attended; null until training is underway |
| `completed` | 0/1, nullable | **Derived by the server**, never entered directly: `attendance_pct >= 60` |
| `placement_status` | text | `Placed` / `Not Placed` / `Unknown` (default) |
| `employer` | text, nullable | Only set when `placement_status = "Placed"` |

## Setup & running

### Option A — plain Python
```bash
git clone <your-repo-url>
cd sih-tracker
python -m venv venv && source venv/bin/activate      # optional
pip install flask python-dotenv pillow
cp .env.example .env                                  # edit SECRET_KEY if you like
python app.py
```
Open http://localhost:5000 — the SQLite database is created and seeded
from `data/sample_data.csv` automatically on first run.

### Option B — Docker
```bash
cp .env.example .env
docker compose up --build
```

### Training the certificate classifier (optional, needs internet)
The web app runs fine without this — the certificate-check feature just
reports itself as unavailable until the model is trained.
```bash
pip install torch torchvision pillow
python ml/generate_images.py       # generates 80 synthetic certificates
python ml/train_classifier.py      # fine-tunes MobileNetV2, ~1-2 min on CPU
```
This prints train/test accuracy and saves `ml/certificate_model.pt`.
**Important:** the 8 certificate templates are split by template (6
train / 2 test), not by individual image — splitting randomly would let
near-duplicate images of the same template appear on both sides and the
reported accuracy would be meaningless.

## How the pieces connect (Task 5)

1. Coordinator fills the enrolment form → server validates every field
   and computes `completed` itself, so every user sees the same number
   for the same data (never trust a derived value from the client).
2. Course dashboard recalculates completion/placement rate live from
   whatever is in the database — no cached numbers to go stale.
3. When updating a candidate's placement, the coordinator can optionally
   upload a certificate photo. If the model is trained, it returns a
   genuine/tampered verdict with a confidence score; below 65%
   confidence it says "uncertain" rather than guessing. If the model
   isn't trained at all, the UI says so and the placement can still be
   saved manually — nothing blocks or breaks.
4. Empty/loading/error states: the candidate list shows "No candidates
   found" instead of an empty table; the certificate checker shows
   "Checking..." then a clear result or a clear failure reason, never a
   blank box.

## Manual check performed (spot-checking one calculated figure)

Using `data/sample_data.csv`: filtering rows with
`course = "Mobile Phone Repair"` gives the enrolled count; among those,
counting `completed = 1` and dividing gives the completion rate shown
on the dashboard for that course — verified by hand against the CSV to
confirm the app's SQL aggregation matches a manual count.

## What's not finished

- The certificate model is trained on synthetic images only; real
  photographed certificates (with lighting/glare/skew) would need a
  larger, real dataset to generalise well — noted here rather than
  overstating the model's real-world accuracy.
- No authentication/login screen — out of scope for an Easy-level,
  2-day assessment focused on the core tracking flow.

## Tech stack

Python 3.11, Flask, SQLite, vanilla HTML/CSS/JS, PyTorch + torchvision
(MobileNetV2 transfer learning, training only — not required to run the
app), Docker, GitHub Actions (build check).
