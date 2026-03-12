"""
============================================================
 FlipLearn – Quiz Dataset Updater
 Author  : Uttam Vitthal Bhise

 SCHEMA (quiz_dataset.csv)
 ──────────────────────────
 One row per student per quiz attempt.

   record_id       – DB_<student_id>_<quiz_id>
   usn             – student roll number
   student_name    – full name
   subject_code    – subject code (DS, PY, …)
   subject_name    – subject full name
   quiz_title      – quiz title
   score           – raw score obtained (from QuizAttempt.score)
   total_marks     – quiz total marks (from Quiz.total_marks)
   score_pct       – score / total_marks * 100 (0–100)
   time_taken_min  – minutes taken to complete the quiz
   attempted_at    – ISO timestamp of attempt
   performance_label – High / Medium / Low / At-Risk based on score_pct

 Thread safety: threading.Lock protects CSV I/O.
============================================================
"""

import pathlib
import threading
import logging

import pandas as pd

logger = logging.getLogger('fliplearn.quiz_dataset')

ML_DIR         = pathlib.Path(__file__).parent
QUIZ_CSV       = ML_DIR / 'quiz_dataset.csv'

_COLUMNS = [
    'record_id', 'usn', 'student_name',
    'subject_code', 'subject_name', 'quiz_title',
    'score', 'total_marks', 'score_pct',
    'time_taken_min', 'attempted_at', 'performance_label',
]

_lock = threading.Lock()


# ── Helpers ─────────────────────────────────────────────────────

def _derive_label(score_pct: float) -> str:
    if score_pct >= 75: return 'High'
    if score_pct >= 50: return 'Medium'
    if score_pct >= 35: return 'Low'
    return 'At-Risk'


def _read() -> pd.DataFrame:
    if QUIZ_CSV.exists():
        try:
            return pd.read_csv(QUIZ_CSV)
        except Exception:
            pass
    return pd.DataFrame(columns=_COLUMNS)


def _write(df: pd.DataFrame) -> None:
    QUIZ_CSV.parent.mkdir(parents=True, exist_ok=True)
    tmp = QUIZ_CSV.with_suffix('.tmp')
    df.to_csv(tmp, index=False)
    tmp.replace(QUIZ_CSV)


# ── Core: upsert one quiz attempt ────────────────────────────────

def upsert_quiz_row(attempt) -> bool:
    """
    Insert or update a row in quiz_dataset.csv for a QuizAttempt instance.
    Returns True if a new row was inserted.
    """
    try:
        student  = attempt.student
        quiz     = attempt.quiz
        subject  = quiz.subject

        try:
            usn = student.student_profile.roll_number
        except Exception:
            usn = f'USN_{student.id}'

        student_name = student.get_full_name().strip() or student.username
        total_marks  = float(quiz.total_marks) if quiz.total_marks else 10.0
        score        = float(attempt.score)
        score_pct    = round((score / total_marks * 100) if total_marks > 0 else 0.0, 2)

        row = {
            'record_id':        f'DB_{student.id}_{quiz.id}',
            'usn':              usn,
            'student_name':     student_name,
            'subject_code':     subject.code,
            'subject_name':     subject.name,
            'quiz_title':       quiz.title,
            'score':            round(score, 2),
            'total_marks':      int(total_marks),
            'score_pct':        score_pct,
            'time_taken_min':   round(float(attempt.time_taken_minutes), 1),
            'attempted_at':     str(attempt.attempted_at),
            'performance_label': _derive_label(score_pct),
        }

        key    = str(row['record_id'])
        is_new = False

        with _lock:
            df = _read()
            if 'record_id' in df.columns and key in df['record_id'].astype(str).values:
                idx = df.index[df['record_id'].astype(str) == key][0]
                for col, val in row.items():
                    df.at[idx, col] = val
                logger.info(f'[QuizDS] Updated {key}')
            else:
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                logger.info(f'[QuizDS] Inserted {key} (total={len(df)})')
                is_new = True
            _write(df)

        return is_new

    except Exception as e:
        logger.error(f'[QuizDS] upsert_quiz_row error: {e}')
        return False


# ── Bulk seed from DB ─────────────────────────────────────────────

def seed_from_db() -> int:
    """
    Pull all QuizAttempt records from the DB and write quiz_dataset.csv.
    Safe to call repeatedly (upserts, not duplicates).
    Returns number of rows written.
    """
    try:
        from flipped_app.models import QuizAttempt
        attempts = QuizAttempt.objects.select_related(
            'student', 'student__student_profile',
            'quiz', 'quiz__subject'
        ).all()
        count = 0
        for attempt in attempts:
            upsert_quiz_row(attempt)
            count += 1
        logger.info(f'[QuizDS] Seeded {count} rows from DB.')
        return count
    except Exception as e:
        logger.error(f'[QuizDS] seed_from_db error: {e}')
        return 0
