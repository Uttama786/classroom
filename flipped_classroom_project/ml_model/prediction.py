"""
============================================================
 Flipped Classroom – ML Prediction Module
 Author  : Uttam Vitthal Bhise
 Description:
   Loads trained models and predicts student performance
   in real-time from the Django database.
============================================================
"""

import os
import numpy as np
import pandas as pd
import joblib

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')

FEATURES = [
    'videos_watched', 'total_video_time_minutes',
    'quiz_avg_score', 'assignment_avg_marks',
    'attendance_percentage', 'participation_score', 'previous_gpa'
]


def _load_models():
    """Load all saved models from disk."""
    scaler  = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    le      = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    rf_reg  = joblib.load(os.path.join(MODELS_DIR, 'rf_regressor.pkl'))
    rf_cls  = joblib.load(os.path.join(MODELS_DIR, 'rf_classifier.pkl'))
    return scaler, le, rf_reg, rf_cls


def predict_student(features_dict: dict, models_tuple=None) -> dict:
    """
    Predict performance for a single student.

    Parameters
    ----------
    features_dict : dict
        Keys matching FEATURES list.

    Returns
    -------
    dict with keys:
        predicted_score  : float
        predicted_label  : str  ('High' / 'Medium' / 'Low' / 'At-Risk')
        is_at_risk       : bool
        confidence       : float  (classification probability)
    """
    if models_tuple is None:
        try:
            scaler, le, rf_reg, rf_cls = _load_models()
        except FileNotFoundError:
            raise RuntimeError(
                "Trained models not found. "
                "Please run ml_model/model_training.py first."
            )
    else:
        scaler, le, rf_reg, rf_cls = models_tuple

    # Use DataFrame so scaler receives correct feature names (avoids sklearn warning)
    feature_values = pd.DataFrame(
        [[features_dict.get(f, 0) for f in FEATURES]],
        columns=FEATURES,
        dtype=float,
    )
    feature_scaled = scaler.transform(feature_values)

    predicted_score = float(np.clip(rf_reg.predict(feature_scaled)[0], 0, 100))
    predicted_cls   = rf_cls.predict(feature_scaled)[0]
    predicted_label = le.inverse_transform([predicted_cls])[0]
    probabilities   = rf_cls.predict_proba(feature_scaled)[0]
    confidence      = float(np.max(probabilities))

    is_at_risk = predicted_label in ('At-Risk', 'Low') or predicted_score < 40

    return {
        'predicted_score': round(predicted_score, 2),
        'predicted_label': predicted_label,
        'is_at_risk': is_at_risk,
        'confidence': round(confidence * 100, 1),
    }


def predict_all_students():
    """
    Run ML prediction for all StudentPerformance records in the database.
    Updates predicted_score, predicted_label, and is_at_risk fields in fast vectorized batch.

    Returns list of result dicts.
    """
    import django
    import sys
    import os

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if not os.environ.get('DJANGO_SETTINGS_MODULE'):
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'flipped_classroom_project.settings')
    try:
        django.setup()
    except RuntimeError:
        pass

    from flipped_app.models import StudentPerformance, Notification

    try:
        scaler, le, rf_reg, rf_cls = _load_models()
    except FileNotFoundError:
        raise RuntimeError(
            "Trained models not found. Run model_training.py first."
        )

    records = list(StudentPerformance.objects.select_related('student', 'subject').all())
    if not records:
        return []

    # Vectorized feature matrix extraction
    df = pd.DataFrame([
        {
            'videos_watched':           perf.videos_watched,
            'total_video_time_minutes': perf.total_video_time_minutes,
            'quiz_avg_score':           perf.quiz_avg_score,
            'assignment_avg_marks':     perf.assignment_avg_marks,
            'attendance_percentage':    perf.attendance_percentage,
            'participation_score':      perf.participation_score,
            'previous_gpa':             perf.previous_gpa,
        }
        for perf in records
    ], columns=FEATURES, dtype=float)

    X_scaled = scaler.transform(df)

    pred_scores = np.clip(rf_reg.predict(X_scaled), 0, 100).round(2)
    pred_cls = rf_cls.predict(X_scaled)
    pred_labels = le.inverse_transform(pred_cls)

    results = []
    notifications_to_create = []
    
    # Pre-fetch existing unread alerts to avoid duplicate notifications
    existing_alerts = set(
        Notification.objects.filter(is_read=False, message__icontains='⚠️ Alert:')
        .values_list('recipient_id', flat=True)
    )

    for i, perf in enumerate(records):
        score = float(pred_scores[i])
        label = str(pred_labels[i])
        at_risk = label in ('At-Risk', 'Low') or score < 40

        perf.predicted_score = score
        perf.predicted_label = label
        perf.is_at_risk = at_risk

        if at_risk and perf.student_id not in existing_alerts:
            notifications_to_create.append(
                Notification(
                    recipient=perf.student,
                    message=(
                        f"⚠️ Alert: Your performance in {perf.subject.name} is below expected level. "
                        f"Predicted score: {score}. "
                        "Please watch more videos and attempt quizzes."
                    )
                )
            )
            existing_alerts.add(perf.student_id)

        results.append({
            'student': perf.student.get_full_name() or perf.student.username,
            'subject': perf.subject.name,
            'predicted_score': score,
            'predicted_label': label,
            'is_at_risk': at_risk,
        })

    # Bulk update all records in 1 transaction
    StudentPerformance.objects.bulk_update(
        records,
        fields=['predicted_score', 'predicted_label', 'is_at_risk'],
        batch_size=1000
    )

    if notifications_to_create:
        Notification.objects.bulk_create(notifications_to_create, batch_size=500, ignore_conflicts=True)

    return results


def get_feature_importance_chart():
    """Return path to feature importance chart."""
    plots_dir = os.path.join(BASE_DIR, 'plots')
    path = os.path.join(plots_dir, 'rf_classification_feature_importance.png')
    return path if os.path.exists(path) else None


def get_model_comparison_chart():
    """Return path to model comparison chart."""
    plots_dir = os.path.join(BASE_DIR, 'plots')
    path = os.path.join(plots_dir, 'model_comparison.png')
    return path if os.path.exists(path) else None


if __name__ == '__main__':
    # Quick test prediction
    test_features = {
        'videos_watched':           5,
        'total_video_time_minutes': 120,
        'quiz_avg_score':           4.0,
        'assignment_avg_marks':     8.5,
        'attendance_percentage':    55,
        'participation_score':      3.5,
        'previous_gpa':             5.5,
    }
    result = predict_student(test_features)
    print("Test Prediction:", result)
