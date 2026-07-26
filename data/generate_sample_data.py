"""
Task 1 - Prepare the Sample Data
=================================
Generates ~100 realistic candidate records for the Skill Training Centre
Enrolment and Placement Tracker.

Field documentation (also copied into README.md):
    candidate_id     : int, unique. Primary key for the candidate.
    name             : str. Candidate's full name.
    course           : str. One of the 5 courses run by the centre.
    batch            : str. Batch code, e.g. "2026-A". Groups candidates
                        who trained together.
    enrolment_date   : str (YYYY-MM-DD). Date the candidate joined.
    attendance_pct    : float 0-100. % of training sessions attended.
                        Can be missing (awkward case) if attendance was
                        never recorded on paper.
    completed        : int, 0 or 1. Whether the candidate finished the
                        course (centre rule: attendance_pct >= 60).
    placement_status : str. One of "Placed", "Not Placed", "Unknown".
                        "Unknown" = follow-up call was never made -
                        this is realistic and must NOT be treated as
                        "Not Placed" when calculating rates.
    employer         : str or empty. Filled only if placement_status
                        == "Placed".

    -> This is also the column the Task 3 model will eventually help
       predict: placement_status.

Deliberately awkward records included (as required by the brief):
    1. A record with a missing attendance_pct value.
    2. Two very similar names (to test search: "Priya Sharma" / "Priya
       Sharman").
    3. One record with no meaningful data in most fields (a bad/junk
       row a coordinator might submit by mistake).
"""
import csv
import random
from datetime import date, timedelta

random.seed(42)

COURSES = [
    "Basic Computer Literacy",
    "Tailoring & Garment Making",
    "Mobile Phone Repair",
    "Beautician & Wellness",
    "Electrical Wiring Basics",
]

FIRST_NAMES = ["Priya", "Anjali", "Ravi", "Suresh", "Meena", "Kavya", "Arun",
               "Divya", "Karthik", "Lakshmi", "Vijay", "Sangeeta", "Manoj",
               "Deepa", "Rajesh", "Pooja", "Ganesh", "Nithya", "Sanjay",
               "Bhavani"]
LAST_NAMES = ["Kumar", "Sharma", "Raj", "Devi", "Prasad", "Nair", "Reddy",
              "Iyer", "Pillai", "Menon"]

EMPLOYERS = ["ABC Electronics", "Sunrise Garments Pvt Ltd",
             "QuickFix Mobile Services", "Glow Beauty Salon",
             "PowerTech Electricals", "Local Self-Employment"]


def random_date(start_year=2025):
    start = date(start_year, 1, 1)
    return start + timedelta(days=random.randint(0, 300))


def make_record(cid):
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    course = random.choice(COURSES)
    batch = f"{random.choice(['2025', '2026'])}-{random.choice('ABC')}"
    enrol = random_date()
    attendance = round(random.uniform(35, 100), 1)
    completed = 1 if attendance >= 60 else 0

    if completed and random.random() < 0.55:
        placement_status = "Placed"
        employer = random.choice(EMPLOYERS)
    elif random.random() < 0.3:
        placement_status = "Unknown"
        employer = ""
    else:
        placement_status = "Not Placed"
        employer = ""

    return [cid, name, course, batch, enrol.isoformat(), attendance,
            completed, placement_status, employer]


def main():
    rows = [make_record(i) for i in range(1, 96)]

    # --- Awkward case 1: missing attendance_pct ---
    rows.append([96, "Ramesh Babu", "Mobile Phone Repair", "2026-B",
                 "2026-02-10", "", "", "Unknown", ""])

    # --- Awkward case 2: two very similar names ---
    rows.append([97, "Priya Sharma", "Beautician & Wellness", "2026-A",
                 "2026-01-15", 82.0, 1, "Placed", "Glow Beauty Salon"])
    rows.append([98, "Priya Sharman", "Beautician & Wellness", "2026-A",
                 "2026-01-16", 78.5, 1, "Not Placed", ""])

    # --- Awkward case 3: junk / near-empty row ---
    rows.append([99, "", "", "", "", "", "", "", ""])

    rows.append(make_record(100))

    with open("sample_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "name", "course", "batch",
                          "enrolment_date", "attendance_pct", "completed",
                          "placement_status", "employer"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} records to sample_data.csv")


if __name__ == "__main__":
    main()
