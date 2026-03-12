"""
============================================================
 FlipLearn – Real-Time Dataset Updater
 Author  : Uttam Vitthal Bhise

 HOW IT WORKS
 ─────────────
 Called automatically by flipped_app/signals.py every time
 a StudentPerformance row is saved (via _update_engagement()
 or teacher grading).

 1. UPSERT  – adds or updates that student's row in dataset.csv
              (match key = student_id + subject code)
 2. RETRAIN – when RETRAIN_EVERY new real rows have been added
              since the last retrain, retrains all ML models
              in a background thread (non-blocking).

 Thread safety: a threading.Lock protects dataset.csv I/O.
============================================================
"""

import pathlib
import threading
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger('fliplearn.realtime_dataset')

# ── Paths ─────────────────────────────────────────────────────
ML_DIR      = pathlib.Path(__file__).parent
DATASET_CSV = ML_DIR / 'dataset.csv'
COUNTER_FILE = ML_DIR / '.new_rows_since_retrain'   # tiny counter file

# ── Config ────────────────────────────────────────────────────
RETRAIN_EVERY = 5       # retrain after every N new real rows added

# ── Thread safety ─────────────────────────────────────────────
_csv_lock      = threading.Lock()
_retrain_lock  = threading.Lock()   # prevent simultaneous retrains


# ═════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════

def _derive_label(score: float) -> str:
    if score >= 75:  return 'High'
    if score >= 50:  return 'Medium'
    if score >= 35:  return 'Low'
    return 'At-Risk'


def _row_from_performance(perf) -> dict:
    """
    Build a dataset row dict from a StudentPerformance ORM instance.
    Uses the same field mapping as generate_dataset.py so the CSV
    schema stays consistent.
    """
    student = perf.student
    subject = perf.subject

    try:
        prev_gpa = student.student_profile.previous_gpa
    except Exception:
        prev_gpa = perf.previous_gpa

    try:
        usn = student.student_profile.roll_number
    except Exception:
        usn = f"USN_{student.id}"

    student_name = student.get_full_name().strip() or student.username

    return {
        'student_id':               f"DB_{student.id}_{subject.code}",
        'usn':                      usn,
        'student_name':             student_name,
        'videos_watched':           int(perf.videos_watched),
        'total_video_time_minutes': round(float(perf.total_video_time_minutes), 1),
        'quiz_avg_score':           round(float(perf.quiz_avg_score), 2),
        'assignment_avg_marks':     round(float(perf.assignment_avg_marks), 2),
        'attendance_percentage':    round(float(perf.attendance_percentage), 1),
        'participation_score':      round(float(perf.participation_score), 1),
        'previous_gpa':             round(float(prev_gpa), 2),
        'final_exam_score':         round(float(perf.final_exam_score), 1),
        'performance_label':        _derive_label(perf.final_exam_score),
        'appended_at':              datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
    }


def _read_csv() -> pd.DataFrame:
    """Read dataset.csv; return empty DataFrame with correct columns if missing."""
    columns = [
        'student_id', 'usn', 'student_name', 'videos_watched',
        'total_video_time_minutes', 'quiz_avg_score', 'assignment_avg_marks',
        'attendance_percentage', 'participation_score', 'previous_gpa',
        'final_exam_score', 'performance_label', 'appended_at'
    ]
    if DATASET_CSV.exists():
        try:
            df = pd.read_csv(DATASET_CSV)
            # Back-fill usn/student_name columns if the CSV pre-dates this feature
            if 'usn' not in df.columns:
                df.insert(1, 'usn', '')
            if 'student_name' not in df.columns:
                df.insert(2, 'student_name', '')
            if 'appended_at' not in df.columns:
                df['appended_at'] = ''
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=columns)


def _write_csv(df: pd.DataFrame) -> None:
    """Atomically write df back to dataset.csv."""
    DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
    tmp = DATASET_CSV.with_suffix('.tmp')
    df.to_csv(tmp, index=False)
    tmp.replace(DATASET_CSV)          # atomic rename


def _get_counter() -> int:
    try:
        return int(COUNTER_FILE.read_text().strip())
    except Exception:
        return 0


def _set_counter(n: int) -> None:
    COUNTER_FILE.write_text(str(n))


# ═════════════════════════════════════════════════════
# CORE: UPSERT ONE ROW
# ═════════════════════════════════════════════════════

def upsert_performance_row(perf) -> bool:
    """
    Insert or update the dataset.csv row for this StudentPerformance.

    Writes to CSV whenever there is ANY meaningful engagement activity
    (quiz taken, video watched, assignment submitted, chat participation)
    even if the teacher has not yet assigned a final_exam_score.

    - Rows with final_exam_score > 0  → true label from _derive_label()
    - Rows with final_exam_score == 0 → label from predicted_label (ML) or 'Pending'

    Retraining is triggered only when rows with a real final_exam_score > 0
    are newly inserted (to avoid training on unconfirmed labels).

    Returns True if a new row was INSERTED (not just updated).
    """
    has_final = float(perf.final_exam_score) > 0
    has_engagement = (
        int(perf.videos_watched) > 0
        or float(perf.quiz_avg_score) > 0
        or float(perf.assignment_avg_marks) > 0
        or float(perf.participation_score) > 0
        or float(perf.total_video_time_minutes) > 0
        or float(perf.attendance_percentage) > 0
    )

    # Skip rows with absolutely no data at all
    if not has_final and not has_engagement:
        return False

    new_row = _row_from_performance(perf)

    # If no final score yet, use the ML predicted label (or 'Pending')
    if not has_final:
        predicted = getattr(perf, 'predicted_label', None) or 'Pending'
        new_row['performance_label'] = predicted

    key    = str(new_row['student_id'])
    is_new = False

    with _csv_lock:
        df = _read_csv()

        if 'student_id' in df.columns and key in df['student_id'].astype(str).values:
            # UPDATE existing row
            idx = df.index[df['student_id'].astype(str) == key][0]
            for col, val in new_row.items():
                df.at[idx, col] = val
            # Always refresh the timestamp on update
            df.at[idx, 'appended_at'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            logger.info(f"[RealTime] Updated row for {key}")
        else:
            # INSERT new row
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            logger.info(f"[RealTime] Inserted new row for {key} "
                        f"(total={len(df)})")
            is_new = True

        _write_csv(df)

    # Only count toward retraining when a confirmed final score is present
    if is_new and has_final:
        count = _get_counter() + 1
        _set_counter(count)
        logger.info(f"[RealTime] New real rows since last retrain: {count}")
        if count >= RETRAIN_EVERY:
            _set_counter(0)
            _trigger_retrain_background()

    return is_new


# ═════════════════════════════════════════════════════
# BACKGROUND RETRAINING
# ═════════════════════════════════════════════════════

def _trigger_retrain_background() -> None:
    """Spawn a daemon thread to retrain ML models without blocking requests."""
    if _retrain_lock.locked():
        logger.info("[RealTime] Retrain already in progress — skipping.")
        return

    t = threading.Thread(target=_retrain_worker, daemon=True,
                         name='fliplearn-retrain')
    t.start()
    logger.info("[RealTime] Background model retrain triggered.")


def _retrain_worker() -> None:
    """Run inside background thread — retrains all saved_models/*.pkl."""
    with _retrain_lock:
        try:
            logger.info("[RealTime] Starting model retrain ...")
            import importlib
            import sys

            # Run model_training.py in the same process namespace
            training_path = str(ML_DIR / 'model_training.py')
            spec = importlib.util.spec_from_file_location(
                'model_training', training_path)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            logger.info("[RealTime] Model retrain complete ✓")
            print("[FlipLearn RealTime] Models retrained successfully ✓")
        except Exception as e:
            logger.error(f"[RealTime] Retrain failed: {e}")
            print(f"[FlipLearn RealTime] Retrain error: {e}")


# ═════════════════════════════════════════════════════
# MANUAL TRIGGER (admin / management command)
# ═════════════════════════════════════════════════════

def force_retrain() -> None:
    """Manually trigger a background retrain (e.g. from Django admin action)."""
    _set_counter(0)
    _trigger_retrain_background()


def dataset_stats() -> dict:
    """Return quick stats about the current dataset.csv."""
    with _csv_lock:
        df = _read_csv()
    real_rows = df['student_id'].astype(str).str.startswith('DB_').sum() \
        if 'student_id' in df.columns else 0
    return {
        'total_rows':   len(df),
        'real_rows':    int(real_rows),
        'synthetic_rows': len(df) - int(real_rows),
        'pending_until_retrain': max(0, RETRAIN_EVERY - _get_counter()),
        'label_dist': df['performance_label'].value_counts().to_dict()
                      if 'performance_label' in df.columns else {},
    }
