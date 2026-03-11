"""
============================================================
 FlipLearn – Dataset Generation Script
 Author  : Uttam Vitthal Bhise
 Program : M.Tech CSE

 DESCRIPTION:
   Generates dataset.csv from REAL platform activity stored
   in the Django database:

   STUDENT ACTIVITY  → VideoWatchHistory, QuizAttempt
   TEACHER ACTIVITY  → AssignmentSubmission (graded marks),
                        participation_score, final_exam_score
                        set via teacher analytics/grading views
   ADMIN ACTIVITY    → Subject configuration, StudentPerformance
                        records created when subjects are assigned

   If the database has fewer than MIN_REAL_RECORDS real
   student-subject pairs, the script generates realistic
   synthetic records to reach TARGET_TOTAL = 200 rows.

   Synthetic records use proper statistical noise so that
   feature-target correlations are in the realistic 0.65–0.85
   range (unlike the old hard-coded CSV which had 0.99).

 USAGE:
   python ml_model/generate_dataset.py
   (run from inside flipped_classroom_project/ folder)

   OR as management command (if placed in management/commands/):
   python manage.py generate_dataset
============================================================
"""

import os
import sys
import pathlib
import numpy as np
import pandas as pd

# ── Django setup ─────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault('DJANGO_SETTINGS_MODULE',
                      'flipped_classroom_project.settings')

import django
try:
    django.setup()
except RuntimeError:
    pass  # already configured

# ── Config ───────────────────────────────────────────────
MIN_REAL_RECORDS = 20     # minimum DB rows before we augment
TARGET_TOTAL     = 200    # final dataset size (real + synthetic)
RANDOM_SEED      = 42
OUTPUT_PATH = pathlib.Path(__file__).parent / 'dataset.csv'

np.random.seed(RANDOM_SEED)


# ═══════════════════════════════════════════════════════════
# 1.  PULL REAL DATA FROM DJANGO DATABASE
# ═══════════════════════════════════════════════════════════

def pull_real_records() -> pd.DataFrame:
    """
    Extract one row per (student, subject) from the live DB.

    Sources:
      STUDENT  — VideoWatchHistory (videos watched, total watch time)
      STUDENT  — QuizAttempt (quiz average score per subject)
      TEACHER  — AssignmentSubmission.marks_obtained (graded by teacher)
      TEACHER  — StudentPerformance.participation_score (teacher-set)
      TEACHER  — StudentPerformance.final_exam_score (teacher-set)
      ADMIN    — Subject records (configured by admin)
      ADMIN    — StudentProfile.previous_gpa, Attendance records
    """
    from flipped_app.models import (
        StudentPerformance, VideoWatchHistory, QuizAttempt,
        AssignmentSubmission, Attendance, StudentProfile, Subject
    )
    from django.db.models import Avg, Sum, Count, Q

    print("[DB] Querying real student-subject records ...")
    records = []
    perf_qs = StudentPerformance.objects.select_related('student', 'subject').all()

    for perf in perf_qs:
        student = perf.student
        subject = perf.subject

        # ── STUDENT ACTIVITY: video engagement ──
        watch_qs = VideoWatchHistory.objects.filter(
            student=student,
            video__subject=subject
        )
        videos_watched       = watch_qs.filter(completed=True).count()
        total_video_minutes  = watch_qs.aggregate(
            t=Sum('watch_duration_minutes'))['t'] or 0.0

        # ── STUDENT ACTIVITY: quiz performance ──
        # Use raw score average (same scale as _update_engagement stores in DB)
        quiz_attempts = QuizAttempt.objects.filter(
            student=student,
            quiz__subject=subject
        )
        if quiz_attempts.exists():
            quiz_avg = float(
                quiz_attempts.aggregate(Avg('score'))['score__avg'] or 0.0
            )
        else:
            quiz_avg = 0.0

        # ── TEACHER ACTIVITY: assignment grading ──
        # Use raw marks_obtained average (same as _update_engagement stores)
        graded_subs = AssignmentSubmission.objects.filter(
            student=student,
            assignment__subject=subject,
            is_graded=True
        )
        if graded_subs.exists():
            assign_avg = float(
                graded_subs.aggregate(Avg('marks_obtained'))['marks_obtained__avg'] or 0.0
            )
        else:
            assign_avg = 0.0

        # ── ADMIN + STUDENT: attendance ──
        attend_qs = Attendance.objects.filter(student=student, subject=subject)
        total_classes  = attend_qs.count()
        present_classes = attend_qs.filter(present=True).count()
        attendance_pct = (present_classes / total_classes * 100) \
            if total_classes > 0 else perf.attendance_percentage

        # ── TEACHER ACTIVITY: participation score ──
        participation = perf.participation_score  # set by teacher via analytics

        # ── ADMIN: previous GPA from StudentProfile ──
        try:
            prev_gpa = student.student_profile.previous_gpa
        except Exception:
            prev_gpa = perf.previous_gpa

        # ── TEACHER/ADMIN: final exam score ──
        final_score = perf.final_exam_score

        # Only include rows where teacher has set final_exam_score
        if final_score <= 0:
            continue

        # Derive correct label from thresholds
        if final_score >= 75:
            label = 'High'
        elif final_score >= 50:
            label = 'Medium'
        elif final_score >= 35:
            label = 'Low'
        else:
            label = 'At-Risk'

        # ── USN and student name ──
        try:
            usn = student.student_profile.roll_number
        except Exception:
            usn = f"USN_{student.id}"

        student_name = student.get_full_name().strip() or student.username

        records.append({
            'student_id':              f"DB_{student.id}_{subject.code}",
            'usn':                     usn,
            'student_name':            student_name,
            'videos_watched':          int(videos_watched),
            'total_video_time_minutes': round(float(total_video_minutes), 1),
            'quiz_avg_score':          round(quiz_avg, 2),
            'assignment_avg_marks':    round(assign_avg, 2),
            'attendance_percentage':   round(float(attendance_pct), 1),
            'participation_score':     round(float(participation), 1),
            'previous_gpa':            round(float(prev_gpa), 2),
            'final_exam_score':        round(float(final_score), 1),
            'performance_label':       label,
            '_source': 'real'
        })

    df = pd.DataFrame(records)
    print(f"[DB] Extracted {len(df)} real records from database.")
    return df


# ═══════════════════════════════════════════════════════════
# 2.  REALISTIC SYNTHETIC DATA GENERATION
# ═══════════════════════════════════════════════════════════

def _noisy(base: np.ndarray, scale: float,
           low: float, high: float) -> np.ndarray:
    """Add Gaussian noise and clip to [low, high]."""
    return np.clip(base + np.random.normal(0, scale, len(base)), low, high)


def generate_synthetic_records(n: int, start_id: int = 1) -> pd.DataFrame:
    """
    Generate n realistic synthetic student records.

    Design principles:
    ─────────────────
    • Three latent student archetypes drive all features:
        "engaged"   (30%) – high videos, high scores
        "average"   (45%) – mid-range across the board
        "struggling" (25%) – low engagement, At-Risk/Low labels

    • Features are derived from a latent 'effort' score plus
      independent noise, giving realistic correlations (0.65–0.85)
      instead of the near-perfect 0.99 of the old CSV.

    • Teacher-influenced fields:
        assignment_avg_marks  – teacher grades (±noise on student effort)
        participation_score   – teacher-observed score (±noise)
        final_exam_score      – teacher-set outcome (derived from effort)

    • Admin-configured context:
        previous_gpa          – reflects institutional entry criteria
        attendance_percentage – shaped by admin-enforced attendance policy
                                  (min 60% to appear in records)
    """
    rng = np.random

    # ── Latent effort score [0,1] per student ──────────────
    archetype = rng.choice(['engaged', 'average', 'struggling'],
                           size=n, p=[0.30, 0.45, 0.25])

    effort = np.where(
        archetype == 'engaged',
        np.clip(rng.beta(6, 2, n), 0.55, 1.0),
        np.where(
            archetype == 'average',
            np.clip(rng.beta(3, 3, n), 0.30, 0.75),
            np.clip(rng.beta(2, 6, n), 0.0, 0.45)
        )
    )

    # ── STUDENT ACTIVITY: video engagement ─────────────────
    # Max 20 videos available; completed depends on effort
    videos_watched = np.clip(
        np.round(effort * 20 + rng.normal(0, 1.5, n)), 0, 20
    ).astype(int)

    # Avg 30 min/video, ±10 min noise per video
    time_per_video = np.clip(rng.normal(28, 8, n), 10, 45)
    total_video_time = np.clip(
        videos_watched * time_per_video + rng.normal(0, 15, n), 0, 600
    ).round(1)

    # ── STUDENT ACTIVITY: quiz performance (0–10 raw score out of 10) ──
    # Matches scale stored by _update_engagement: avg of QuizAttempt.score
    # Default Quiz.total_marks = 10
    quiz_avg = _noisy(effort * 9.5 + 0.3, 0.9, 1.5, 10.0).round(2)

    # ── TEACHER ACTIVITY: graded assignment marks (0–20 raw marks) ──
    # Matches scale stored by _update_engagement: avg of marks_obtained
    # Default Assignment.total_marks = 20; teacher grades with leniency noise
    assignment_effort_component = effort * 18.5 + 0.5
    teacher_grading_noise       = rng.normal(0, 1.5, n)   # teacher leniency variance
    assignment_avg = _noisy(
        assignment_effort_component + teacher_grading_noise, 1.2, 2.0, 20.0
    ).round(2)

    # ── ADMIN-ENFORCED: attendance policy ──────────────────
    # Admin requires min 60% attendance; high-effort students attend more
    attendance = _noisy(effort * 80 + 18, 8, 35, 99).round(1)

    # ── TEACHER ACTIVITY: participation score ──────────────
    # Teacher-observed in-class; 0–10 scale, correlated with effort
    participation = _noisy(effort * 9.0 + 0.5, 0.9, 0.5, 10.0).round(1)

    # ── ADMIN: previous GPA (institutional entry GPA range 3.5–10) ─
    gpa_base    = effort * 5.5 + 3.8
    previous_gpa = _noisy(gpa_base, 0.8, 3.5, 10.0).round(2)

    # ── TEACHER/ADMIN: final exam score (0–100) ───────────────
    # Normalise each feature to a 0–100 space before weighting
    exam_base = (
        0.28 * (quiz_avg / 10.0 * 100) +          # quiz out of 10 → %
        0.22 * (assignment_avg / 20.0 * 100) +    # assignment out of 20 → %
        0.18 * attendance +                        # already 0–100
        0.15 * (participation / 10.0 * 100) +     # 0–10 → %
        0.12 * (videos_watched / 20.0 * 100) +    # 0–20 → %
        0.05 * (previous_gpa / 10.0 * 100)        # 0–10 GPA → %
    )
    # Add exam-day noise (±5 to ±12 marks, depending on archetype)
    exam_noise_scale = np.where(archetype == 'struggling', 7.0,
                       np.where(archetype == 'average', 5.5, 4.0))
    final_exam_score = np.clip(
        exam_base + rng.normal(0, exam_noise_scale, n), 5, 98
    ).round(1)

    # ── TEACHER/ADMIN: performance label from thresholds ───
    labels = np.where(final_exam_score >= 75, 'High',
             np.where(final_exam_score >= 50, 'Medium',
             np.where(final_exam_score >= 35, 'Low', 'At-Risk')))

    rows = []
    for i in range(n):
        rows.append({
            'student_id':               f"SYN_{start_id + i:04d}",
            'usn':                      f"SYN_{start_id + i:04d}",
            'student_name':             f"Synthetic Student {start_id + i}",
            'videos_watched':           int(videos_watched[i]),
            'total_video_time_minutes': float(total_video_time[i]),
            'quiz_avg_score':           float(quiz_avg[i]),
            'assignment_avg_marks':     float(assignment_avg[i]),
            'attendance_percentage':    float(attendance[i]),
            'participation_score':      float(participation[i]),
            'previous_gpa':             float(previous_gpa[i]),
            'final_exam_score':         float(final_exam_score[i]),
            'performance_label':        str(labels[i]),
            '_source':                  'synthetic',
        })

    df = pd.DataFrame(rows)
    print(f"[SYN] Generated {len(df)} synthetic records "
          f"(engaged={int((archetype=='engaged').sum())}, "
          f"average={int((archetype=='average').sum())}, "
          f"struggling={int((archetype=='struggling').sum())})")
    return df


# ═══════════════════════════════════════════════════════════
# 3.  COMBINE, VALIDATE, AND SAVE
# ═══════════════════════════════════════════════════════════

def validate_and_report(df: pd.DataFrame, n_real: int = 0, n_synthetic: int = 0) -> None:
    """Print quality checks on the final dataset."""
    features = [
        'videos_watched', 'total_video_time_minutes', 'quiz_avg_score',
        'assignment_avg_marks', 'attendance_percentage',
        'participation_score', 'previous_gpa'
    ]
    print("\n" + "=" * 60)
    print("DATASET QUALITY REPORT")
    print("=" * 60)
    print(f"Total rows     : {len(df)}")
    print(f"Real DB rows   : {n_real}")
    print(f"Synthetic rows : {n_synthetic}")
    print(f"\nClass distribution:")
    print(df['performance_label'].value_counts().to_string())

    # Label consistency
    def expected(s):
        if s >= 75: return 'High'
        elif s >= 50: return 'Medium'
        elif s >= 35: return 'Low'
        return 'At-Risk'
    mismatches = (df['performance_label'] != df['final_exam_score'].apply(expected)).sum()
    print(f"\nLabel mismatches (score vs label): {mismatches}  "
          f"{'✔  CLEAN' if mismatches == 0 else '✗  FIX NEEDED'}")

    print(f"\nCorrelations with final_exam_score:")
    corrs = df[features + ['final_exam_score']].corr()['final_exam_score'].drop('final_exam_score')
    for feat, corr in corrs.items():
        bar = '█' * int(abs(corr) * 20)
        print(f"  {feat:<30} {corr:+.3f}  {bar}")

    print(f"\nDescriptive stats:")
    print(df[features + ['final_exam_score']].describe().round(2).to_string())


def generate_dataset(
        target_total: int = TARGET_TOTAL,
        min_real: int = MIN_REAL_RECORDS
) -> None:
    print("=" * 60)
    print("FlipLearn — Dataset Generation")
    print("Sources: Student activity · Teacher grades · Admin data")
    print("=" * 60)

    # Step 1: Pull real DB records
    real_df = pull_real_records()
    n_real  = len(real_df)

    # Step 2: Augment with synthetic records if needed
    n_synthetic = max(0, target_total - n_real)
    if n_real < min_real:
        print(f"[INFO] Only {n_real} real records (min={min_real}). "
              f"Generating {n_synthetic} synthetic records to reach {target_total}.")
    else:
        n_synthetic = max(0, target_total - n_real)
        print(f"[INFO] {n_real} real records found. "
              f"Adding {n_synthetic} synthetic records for total of {target_total}.")

    frames = []
    if n_real > 0:
        frames.append(real_df)

    if n_synthetic > 0:
        syn_df = generate_synthetic_records(n_synthetic, start_id=n_real + 1)
        frames.append(syn_df)

    df = pd.concat(frames, ignore_index=True) if frames else generate_synthetic_records(target_total)

    # Step 3: Fix any label mismatches (ensure label always matches score)
    def correct_label(row):
        s = row['final_exam_score']
        if s >= 75: return 'High'
        elif s >= 50: return 'Medium'
        elif s >= 35: return 'Low'
        return 'At-Risk'

    df['performance_label'] = df.apply(correct_label, axis=1)

    # Step 4: Clean up internal column; preserve real student_ids, assign for synthetic
    df = df.drop(columns=['_source'], errors='ignore')

    # Step 5: Reorder columns to match training script expectation
    df = df[[
        'student_id', 'usn', 'student_name', 'videos_watched',
        'total_video_time_minutes', 'quiz_avg_score', 'assignment_avg_marks',
        'attendance_percentage', 'participation_score', 'previous_gpa',
        'final_exam_score', 'performance_label'
    ]]

    # Step 6: Validate and report
    validate_and_report(df, n_real=n_real, n_synthetic=n_synthetic)

    # Step 7: Save
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Dataset saved to: {OUTPUT_PATH}")
    print(f"   Rows: {len(df)}  |  Columns: {len(df.columns)}")


# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate FlipLearn ML dataset from DB activity + synthetic augmentation'
    )
    parser.add_argument('--total', type=int, default=TARGET_TOTAL,
                        help=f'Target total rows (default: {TARGET_TOTAL})')
    parser.add_argument('--min-real', type=int, default=MIN_REAL_RECORDS,
                        help=f'Min real DB rows before augmenting (default: {MIN_REAL_RECORDS})')
    args = parser.parse_args()
    generate_dataset(target_total=args.total, min_real=args.min_real)
