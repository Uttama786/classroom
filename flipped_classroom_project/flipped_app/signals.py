"""
============================================================
 FlipLearn – Django Signals for Real-Time Dataset Updates
 Author  : Uttam Vitthal Bhise

 SIGNALS WIRED:
   StudentPerformance.post_save
       → upsert_performance_row()   (ml_model/realtime_dataset.py)
       → retrain models in background when RETRAIN_EVERY rows added

 These signals fire automatically inside the same Django process
 whenever _update_engagement() writes to StudentPerformance
 (video watch, quiz attempt, assignment submit/grade) or a
 teacher manually saves a final_exam_score.
============================================================
"""

import logging
from django.db.models.signals import post_save
from django.dispatch import receiver

logger = logging.getLogger('fliplearn.signals')


@receiver(post_save, sender='flipped_app.StudentPerformance')
def on_performance_saved(sender, instance, created, **kwargs):
    """
    Fires every time a StudentPerformance row is created or updated.

    Writes to dataset.csv whenever there is any meaningful engagement
    (quiz taken, video watched, assignment submitted, chat message).
    Rows without a final_exam_score use the ML predicted_label as their
    performance_label.  Model retraining is only triggered for rows
    that have a confirmed final_exam_score > 0.
    """
    try:
        from ml_model.realtime_dataset import upsert_performance_row
        is_new = upsert_performance_row(instance)

        action = "inserted" if is_new else "updated"
        logger.debug(
            f"[Signal] StudentPerformance {action} for "
            f"{instance.student.username} / {instance.subject.code} "
            f"(final_score={instance.final_exam_score})"
        )
    except Exception as e:
        # Never let a signal crash a user request
        logger.error(f"[Signal] on_performance_saved error: {e}")


@receiver(post_save, sender='flipped_app.QuizAttempt')
def on_quiz_attempt_saved(sender, instance, created, **kwargs):
    """
    After a quiz attempt is saved, _update_engagement() has already
    been called by the view (updating StudentPerformance).  This signal
    acts as a safety net: if the StudentPerformance row was updated via
    a path that didn't call _update_engagement explicitly
    (e.g. direct DB write / admin panel), we re-snapshot it here.
    """
    try:
        from flipped_app.models import StudentPerformance
        from ml_model.realtime_dataset import upsert_performance_row
        perf_qs = StudentPerformance.objects.filter(
            student=instance.student,
            subject=instance.quiz.subject
        )
        if perf_qs.exists():
            upsert_performance_row(perf_qs.first())
    except Exception as e:
        logger.error(f"[Signal] on_quiz_attempt_saved error: {e}")


@receiver(post_save, sender='flipped_app.AssignmentSubmission')
def on_submission_graded(sender, instance, created, **kwargs):
    """
    Fires when a teacher grades an AssignmentSubmission (is_graded=True).
    Snapshots the updated StudentPerformance into dataset.csv.
    """
    if not instance.is_graded:
        return
    try:
        from flipped_app.models import StudentPerformance
        from ml_model.realtime_dataset import upsert_performance_row
        perf_qs = StudentPerformance.objects.filter(
            student=instance.student,
            subject=instance.assignment.subject
        )
        if perf_qs.exists():
            upsert_performance_row(perf_qs.first())
    except Exception as e:
        logger.error(f"[Signal] on_submission_graded error: {e}")


@receiver(post_save, sender='flipped_app.VideoWatchHistory')
def on_video_watched(sender, instance, created, **kwargs):
    """
    Fires when a VideoWatchHistory record is saved (student finishes a video).
    Snapshots the updated StudentPerformance row so video engagement
    is reflected in the CSV immediately.
    """
    if not instance.completed:
        return
    try:
        from flipped_app.models import StudentPerformance
        from ml_model.realtime_dataset import upsert_performance_row
        perf_qs = StudentPerformance.objects.filter(
            student=instance.student,
            subject=instance.video.subject
        )
        if perf_qs.exists():
            upsert_performance_row(perf_qs.first())
    except Exception as e:
        logger.error(f"[Signal] on_video_watched error: {e}")
