import os
import re
from datetime import datetime, date
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

from db import init_db, seed_from_csv, get_conn

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")

COURSES = [
    "Basic Computer Literacy",
    "Tailoring & Garment Making",
    "Mobile Phone Repair",
    "Beautician & Wellness",
    "Electrical Wiring Basics",
]

BASE_DIR = os.path.dirname(__file__)


def ensure_db():
    init_db()
    csv_path = os.path.join(BASE_DIR, "data", "sample_data.csv")
    if os.path.exists(csv_path):
        seed_from_csv(csv_path)


# ---------------------------------------------------------------- helpers --
def validate_candidate(form):
    """Server-side validation. Returns (cleaned_dict, list_of_errors)."""
    errors = []
    name = (form.get("name") or "").strip()
    course = (form.get("course") or "").strip()
    batch = (form.get("batch") or "").strip()
    enrolment_date = (form.get("enrolment_date") or "").strip()
    attendance_raw = (form.get("attendance_pct") or "").strip()

    if not name:
        errors.append("Name is required.")
    if course not in COURSES:
        errors.append("Please choose a valid course.")
    if not batch:
        errors.append("Batch is required.")

    try:
        d = datetime.strptime(enrolment_date, "%Y-%m-%d").date()
        if d > date.today():
            errors.append("Enrolment date cannot be in the future.")
    except ValueError:
        errors.append("Enrolment date must be a valid date (YYYY-MM-DD).")

    attendance_pct = None
    if attendance_raw != "":
        try:
            attendance_pct = float(attendance_raw)
            if not (0 <= attendance_pct <= 100):
                errors.append("Attendance % must be between 0 and 100.")
        except ValueError:
            errors.append("Attendance % must be a number.")

    # completed is DERIVED on the server, never trusted from the client,
    # so every user sees the same rule applied the same way.
    completed = None
    if attendance_pct is not None:
        completed = 1 if attendance_pct >= 60 else 0

    return {
        "name": name, "course": course, "batch": batch,
        "enrolment_date": enrolment_date, "attendance_pct": attendance_pct,
        "completed": completed,
    }, errors


# ------------------------------------------------------------------ views --
@app.route("/")
def dashboard():
    conn = get_conn()
    rows = conn.execute(
        """SELECT course,
                  COUNT(*) AS enrolled,
                  SUM(CASE WHEN completed = 1 THEN 1 ELSE 0 END) AS completed,
                  SUM(CASE WHEN placement_status = 'Placed' THEN 1 ELSE 0 END) AS placed
           FROM candidates GROUP BY course"""
    ).fetchall()
    conn.close()

    summary = []
    for r in rows:
        enrolled = r["enrolled"] or 0
        completed = r["completed"] or 0
        placed = r["placed"] or 0
        completion_rate = round(100 * completed / enrolled, 1) if enrolled else None
        # Placement rate is defined here as placed / completed (of those who
        # finished, how many got jobs) - documented in README so it isn't
        # ambiguous. Candidates never completed are excluded from this rate.
        placement_rate = round(100 * placed / completed, 1) if completed else None
        summary.append({
            "course": r["course"], "enrolled": enrolled, "completed": completed,
            "placed": placed, "completion_rate": completion_rate,
            "placement_rate": placement_rate,
        })

    # Weak course = lowest placement rate among courses with data.
    scored = [s for s in summary if s["placement_rate"] is not None]
    weakest = min(scored, key=lambda s: s["placement_rate"]) if scored else None

    return render_template("dashboard.html", summary=summary, weakest=weakest)


@app.route("/candidates")
def candidate_list():
    q = (request.args.get("q") or "").strip()
    conn = get_conn()
    if q:
        like = f"%{q}%"
        rows = conn.execute(
            "SELECT * FROM candidates WHERE name LIKE ? ORDER BY candidate_id DESC",
            (like,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM candidates ORDER BY candidate_id DESC"
        ).fetchall()
    conn.close()
    return render_template("candidates.html", candidates=rows, q=q)


@app.route("/candidates/new", methods=["GET", "POST"])
def new_candidate():
    if request.method == "POST":
        cleaned, errors = validate_candidate(request.form)
        if errors:
            for e in errors:
                flash(e, "error")
            return render_template("new_candidate.html", courses=COURSES, form=request.form)

        conn = get_conn()
        conn.execute(
            """INSERT INTO candidates
               (name, course, batch, enrolment_date, attendance_pct,
                completed, placement_status, employer)
               VALUES (?,?,?,?,?,?,?,?)""",
            (cleaned["name"], cleaned["course"], cleaned["batch"],
             cleaned["enrolment_date"], cleaned["attendance_pct"],
             cleaned["completed"], "Unknown", ""),
        )
        conn.commit()
        conn.close()
        flash("Candidate saved.", "success")
        return redirect(url_for("candidate_list"))

    return render_template("new_candidate.html", courses=COURSES, form={})


@app.route("/candidates/<int:candidate_id>/placement", methods=["GET", "POST"])
def update_placement(candidate_id):
    conn = get_conn()
    candidate = conn.execute(
        "SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,)
    ).fetchone()
    if candidate is None:
        conn.close()
        flash("Candidate not found.", "error")
        return redirect(url_for("candidate_list"))

    if request.method == "POST":
        status = request.form.get("placement_status")
        employer = (request.form.get("employer") or "").strip()
        if status not in ("Placed", "Not Placed", "Unknown"):
            flash("Invalid placement status.", "error")
        else:
            if status != "Placed":
                employer = ""
            conn.execute(
                "UPDATE candidates SET placement_status = ?, employer = ? WHERE candidate_id = ?",
                (status, employer, candidate_id),
            )
            conn.commit()
            conn.close()
            flash("Placement updated.", "success")
            return redirect(url_for("candidate_list"))

    conn.close()
    return render_template("update_placement.html", candidate=candidate)


@app.route("/api/verify-certificate", methods=["POST"])
def verify_certificate():
    """Task 5 integration point: a coordinator confirming a placement can
    upload a photo of the candidate's completion certificate and get a
    genuine/tampered/uncertain hint before saving. Falls back gracefully
    (available: false) if the model hasn't been trained - see
    ml/train_classifier.py. Never forces a verdict it isn't confident in
    (see CONFIDENCE_FLOOR in ml/predict.py)."""
    try:
        from ml.predict import verify_certificate as run_model
    except Exception:
        return jsonify({"available": False,
                         "reason": "model not trained yet - run ml/train_classifier.py"})

    file = request.files.get("certificate")
    if not file or file.filename == "":
        return jsonify({"available": False, "reason": "no file uploaded"}), 400

    try:
        result = run_model(file.read())
    except Exception as e:
        return jsonify({"available": False, "reason": f"could not process image: {e}"}), 400

    result["available"] = True
    return jsonify(result)


if __name__ == "__main__":
    ensure_db()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
else:
    ensure_db()
