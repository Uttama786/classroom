"""
generate_dummy_120.py
──────────────────────────────────────────────────────────────────────────────
Generates a fresh dataset.csv with exactly 120 BE 6th-semester students:
  • 60 CS  branch  →  USN  2SD22CS001 – 2SD22CS060
  • 60 IS  branch  →  USN  2SD22IS001 – 2SD22IS060

Each student gets ONE representative performance record – one row per student
(the subject is kept consistent per student so the dataset has exactly 120 rows).

Realistic correlations:
  previous_gpa  →  attendance  →  quiz  →  assignment  →  final_exam_score

Author : Uttam Vitthal Bhise
"""

import csv, random, os
from datetime import datetime, timedelta

random.seed(99)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")

HEADER = [
    "student_id", "usn", "student_name",
    "videos_watched", "total_video_time_minutes",
    "quiz_avg_score", "assignment_avg_marks",
    "attendance_percentage", "participation_score",
    "previous_gpa", "final_exam_score",
    "performance_label", "appended_at",
]

SUBJECTS = ["DS", "PY", "WD", "CN", "DSC", "AIML"]

# ── 120 Indian names (60 CS + 60 IS) ─────────────────────────────────────────
CS_NAMES = [
    "Aarav Patil", "Abhishek Kulkarni", "Aditi Rao", "Aditya Nair", "Akash Joshi",
    "Akshay Reddy", "Amisha Sharma", "Amrita Verma", "Ananya Iyer", "Anil Desai",
    "Anirudh Bhosale", "Anjali Menon", "Ankit Singh", "Anuja Pillai", "Arjun Kumar",
    "Aryan Mehta", "Ashish Gupta", "Ashwini Tiwari", "Bhavana Naik", "Chirag Patel",
    "Deepak Shinde", "Deepika Kamath", "Devika Hegde", "Dhruv Shetty", "Dinesh More",
    "Divya Chavan", "Ganesh Jadhav", "Gayatri Pawar", "Harish Kamble", "Harsha Gowda",
    "Hemanth Rao", "Hrishikesh Jain", "Ishaan Pandey", "Ishita Deshpande", "Jayesh Naik",
    "Jyoti Patil", "Karan Shinde", "Kartik Sawant", "Kavita Bhatt", "Kedar Kulkarni",
    "Kiran Wagh", "Kishore Nair", "Komal Mane", "Krishna Pillai", "Lakshmi Reddy",
    "Mahesh Kale", "Manasi Shirke", "Manish Tupe", "Mayur Ghule", "Meera Padte",
    "Mihir Sonar", "Milind Patkar", "Minal Deore", "Mohan Gaikwad", "Nagesh Bhide",
    "Ganesh Salunkhe", "Seema Landge", "Vinayak Powar", "Supriya Bongane", "Nikhil Ingule",
]

IS_NAMES = [
    "Omkar Dhole", "Pallavi Sawant", "Parth Salvi", "Pooja Kumbhar", "Prajwal Patil",
    "Prajakta Gholap", "Pranav Rokade", "Pranoti Mhatre", "Pratik Zaware", "Preeti Chaudhari",
    "Priya Gavhane", "Priyanka Thorat", "Pushkar Kharat", "Rahul Ambekar", "Rajan Bagul",
    "Rajeshwari Pol", "Rakesh Nimbalkar", "Ramesh Kshirsagar", "Rashmi Waghmare", "Ravi Bankar",
    "Riddhi Sanas", "Ritesh Hinge", "Rohini Korde", "Rohit Deshpande", "Rupesh Mule",
    "Rutuja Pawar", "Sachin Chate", "Sagun Surve", "Sahil Khilare", "Sakshi Wakade",
    "Sameer Gite", "Sangita Bhor", "Sanjay Dalvi", "Sanket Bansode", "Sarika Garud",
    "Saurabh Shewale", "Sayali Rathod", "Shilpa Valange", "Shiv Nikalje", "Shradhha Panchal",
    "Shubham Ghadge", "Shweta Sonawane", "Siddhanth Pund", "Sonal Mankar", "Sonam Ghode",
    "Suhas Talekar", "Sumit Shete", "Supriya Kolhe", "Suraj Jagtap", "Sushant Bodke",
    "Swapnil Barde", "Tejal Narkar", "Tejas Ugale", "Tushar Katkar", "Uday Kambale",
    "Ujwala Barhate", "Utkarsh Nage", "Vaibhav Dakhore", "Varsha Chavan", "Vikram Shelke",
]

# ── Archetypes: (gpa_lo, gpa_hi, att_lo, att_hi, quiz_pct, asgn_pct, part, vids, score_lo, score_hi)
ARCHETYPES = {
    "Distinction":  dict(gpa=(8.5,10.0), att=(88,100), quiz=(0.80,1.00), asgn=(0.82,1.00),
                         part=(7.5,10.0), vids=(4,10), score=(80,100)),
    "First_Class":  dict(gpa=(7.0, 8.4), att=(75, 94), quiz=(0.60,0.82), asgn=(0.62,0.85),
                         part=(5.5, 8.0), vids=(2,7),  score=(60, 80)),
    "Pass":         dict(gpa=(5.5, 7.0), att=(55, 78), quiz=(0.40,0.62), asgn=(0.42,0.65),
                         part=(3.5, 6.0), vids=(1,5),  score=(45, 62)),
    "Low":          dict(gpa=(4.0, 5.8), att=(35, 62), quiz=(0.15,0.42), asgn=(0.18,0.45),
                         part=(1.5, 4.0), vids=(0,4),  score=(28, 48)),
    "AtRisk":       dict(gpa=(3.0, 5.0), att=(15, 45), quiz=(0.00,0.28), asgn=(0.05,0.32),
                         part=(0.5, 3.0), vids=(0,3),  score=(5,  35)),
}

# ── Distribution across 120 students ──────────────────────────────────────────
ARCHETYPE_POOL = (
    ["Distinction"] * 30 +   # 25 %
    ["First_Class"] * 42 +   # 35 %
    ["Pass"]        * 28 +   # 23 %
    ["Low"]         * 12 +   # 10 %
    ["AtRisk"]      *  8     #  7 %
)
random.shuffle(ARCHETYPE_POOL)


def clamp(v, lo, hi):   return max(lo, min(hi, v))
def rnd(v, d=1):        return round(v, d)

def label(score):
    if score >= 75: return "High"
    if score >= 50: return "Medium"
    if score >= 35: return "Low"
    return "At-Risk"

def rand_ts():
    base  = datetime(2026, 1, 5, 8, 0, 0)
    delta = timedelta(days=random.randint(0, 82),
                      hours=random.randint(0, 14),
                      minutes=random.randint(0, 59))
    return (base + delta).strftime("%Y-%m-%d %H:%M:%S IST")

QUIZ_MAX  = 10
ASGN_MAX  = 30
VID_TIME  = {"DS":60,"PY":55,"WD":65,"CN":50,"DSC":70,"AIML":75}   # avg max mins


def make_row(student_num: int, usn: str, name: str, branch: str, arch_name: str) -> dict:
    a = ARCHETYPES[arch_name]
    subj = SUBJECTS[student_num % len(SUBJECTS)]          # rotate subjects evenly

    # Per-student fixed values
    gpa  = rnd(clamp(random.uniform(*a["gpa"])  + random.gauss(0, 0.08), 0, 10))
    att  = rnd(clamp(random.uniform(*a["att"])  + random.gauss(0, 2.0),  0, 100))
    quiz = rnd(clamp(random.uniform(*a["quiz"]) * QUIZ_MAX + random.gauss(0, 0.2), 0, QUIZ_MAX))
    asgn = rnd(clamp(random.uniform(*a["asgn"]) * ASGN_MAX + random.gauss(0, 0.4), 0, ASGN_MAX))
    part = rnd(clamp(random.uniform(*a["part"]) + random.gauss(0, 0.3), 0, 10))
    vids = clamp(random.randint(*a["vids"]), 0, 15)
    vtime = rnd(clamp(vids * random.uniform(6, VID_TIME[subj] / max(vids, 1))
                      + random.gauss(0, 3), 0, 300))

    # Final score — correlated composite
    raw = (gpa/10*35 + att/100*25 + quiz/QUIZ_MAX*20 + asgn/ASGN_MAX*15 + part/10*5)
    score = rnd(clamp(raw + random.gauss(0, 2), a["score"][0]-3, a["score"][1]+3))

    sid = f"DB_{branch}_{student_num:03d}_{subj}"

    return {
        "student_id":               sid,
        "usn":                      usn,
        "student_name":             name,
        "videos_watched":           vids,
        "total_video_time_minutes": vtime,
        "quiz_avg_score":           quiz,
        "assignment_avg_marks":     asgn,
        "attendance_percentage":    att,
        "participation_score":      part,
        "previous_gpa":             gpa,
        "final_exam_score":         score,
        "performance_label":        label(score),
        "appended_at":              rand_ts(),
    }


def main():
    rows = []
    arch_idx = 0

    # ── 60 CS students ────────────────────────────────────────────────────────
    for i, name in enumerate(CS_NAMES, start=1):
        usn      = f"2SD22CS{i:03d}"
        arch     = ARCHETYPE_POOL[arch_idx]; arch_idx += 1
        rows.append(make_row(i, usn, name, "CS", arch))

    # ── 60 IS students ────────────────────────────────────────────────────────
    for i, name in enumerate(IS_NAMES, start=1):
        usn      = f"2SD22IS{i:03d}"
        arch     = ARCHETYPE_POOL[arch_idx]; arch_idx += 1
        rows.append(make_row(i, usn, name, "IS", arch))

    # ── Overwrite dataset.csv completely ──────────────────────────────────────
    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    labels = [r["performance_label"] for r in rows]
    scores = [r["final_exam_score"]  for r in rows]
    gpas   = [r["previous_gpa"]      for r in rows]

    print(f"\n{'='*58}")
    print(f"  FlipLearn — BE 6th Sem Dummy Dataset Generator")
    print(f"{'='*58}")
    print(f"  Total students  : {len(rows)}")
    print(f"  CS branch       : 60  (2SD22CS001 – 2SD22CS060)")
    print(f"  IS branch       : 60  (2SD22IS001 – 2SD22IS060)")
    print(f"  Saved to        : {DATASET_PATH}")
    print(f"\n  📊 Class Distribution:")
    for lbl in ["High", "Medium", "Low", "At-Risk"]:
        n = labels.count(lbl)
        bar = "█" * n
        pct = n / len(rows) * 100
        print(f"     {lbl:<10} {n:>3} ({pct:4.1f}%)  {bar}")
    print(f"\n  📈 Score  — Mean: {sum(scores)/len(scores):.1f} "
          f"| Min: {min(scores)} | Max: {max(scores)}")
    print(f"  🎓 GPA    — Mean: {sum(gpas)/len(gpas):.2f}   "
          f"| Min: {min(gpas)} | Max: {max(gpas)}")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
