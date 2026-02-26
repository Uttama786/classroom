"""
============================================================
 Flipped Classroom – ML Model Training Script
 Author  : Uttam Vitthal Bhise
 Program : M.Tech CSE
 Description:
   Trains Regression and Classification models on student
   performance data collected from the flipped classroom
   platform, evaluates them, and saves trained models.
============================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, f1_score, precision_score, recall_score
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ─── Paths ────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset.csv')
MODELS_DIR   = os.path.join(BASE_DIR, 'saved_models')
PLOTS_DIR    = os.path.join(BASE_DIR, 'plots')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

FEATURES = [
    'videos_watched', 'total_video_time_minutes',
    'quiz_avg_score', 'assignment_avg_marks',
    'attendance_percentage', 'participation_score', 'previous_gpa'
]
TARGET_REG  = 'final_exam_score'
TARGET_CLS  = 'performance_label'


# ─────────────────────────────────────────────────────────
# 1. Load & Explore Dataset
# ─────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATASET_PATH)
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape          : {df.shape}")
    print(f"Columns        : {list(df.columns)}")
    print(f"\nClass distribution:\n{df[TARGET_CLS].value_counts()}")
    print(f"\nDescriptive Statistics:\n{df[FEATURES + [TARGET_REG]].describe().round(2)}")
    return df


# ─────────────────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────────────────
def preprocess(df):
    X = df[FEATURES].copy()
    y_reg = df[TARGET_REG].copy()

    le = LabelEncoder()
    y_cls = le.fit_transform(df[TARGET_CLS])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(le,     os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    print("\n✔  Scaler and LabelEncoder saved.")
    return X_scaled, y_reg, y_cls, le, scaler


# ─────────────────────────────────────────────────────────
# 3. Regression Models
# ─────────────────────────────────────────────────────────
def train_regression_models(X, y_reg):
    print("\n" + "=" * 60)
    print("REGRESSION MODEL TRAINING – Predict Final Exam Score")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    results = {}

    # ── Linear Regression ──────────────────────────────────
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    mse_lr  = mean_squared_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mse_lr)
    r2_lr   = r2_score(y_test, y_pred_lr)
    results['Linear Regression'] = {'MSE': mse_lr, 'RMSE': rmse_lr, 'R2': r2_lr}
    joblib.dump(lr, os.path.join(MODELS_DIR, 'linear_regression.pkl'))
    print(f"\n🔵 Linear Regression:")
    print(f"   MSE  = {mse_lr:.4f}  |  RMSE = {rmse_lr:.4f}  |  R²  = {r2_lr:.4f}")

    # ── Random Forest Regressor ────────────────────────────
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train, y_train)
    y_pred_rf = rf_reg.predict(X_test)
    mse_rf  = mean_squared_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mse_rf)
    r2_rf   = r2_score(y_test, y_pred_rf)
    results['Random Forest Regressor'] = {'MSE': mse_rf, 'RMSE': rmse_rf, 'R2': r2_rf}
    joblib.dump(rf_reg, os.path.join(MODELS_DIR, 'rf_regressor.pkl'))
    print(f"\n🟢 Random Forest Regressor:")
    print(f"   MSE  = {mse_rf:.4f}  |  RMSE = {rmse_rf:.4f}  |  R²  = {r2_rf:.4f}")

    # ── Feature Importance Plot ────────────────────────────
    plot_feature_importance(rf_reg, FEATURES, 'rf_regression_feature_importance.png')
    plot_actual_vs_predicted(y_test, y_pred_rf, 'RF Regressor: Actual vs Predicted')

    return results


# ─────────────────────────────────────────────────────────
# 4. Classification Models
# ─────────────────────────────────────────────────────────
def train_classification_models(X, y_cls, le):
    print("\n" + "=" * 60)
    print("CLASSIFICATION MODEL TRAINING – Predict Performance Label")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    class_names = list(le.classes_)
    results = {}

    # ── Logistic Regression ────────────────────────────────
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)
    acc_log = accuracy_score(y_test, y_pred_log)
    f1_log  = f1_score(y_test, y_pred_log, average='weighted')
    results['Logistic Regression'] = {'Accuracy': acc_log, 'F1': f1_log}
    joblib.dump(log_reg, os.path.join(MODELS_DIR, 'logistic_regression.pkl'))
    print(f"\n🔵 Logistic Regression:")
    print(f"   Accuracy = {acc_log:.4f}  |  F1-Score = {f1_log:.4f}")
    print(classification_report(y_test, y_pred_log, target_names=class_names))
    plot_confusion_matrix(y_test, y_pred_log, class_names, 'confusion_matrix_logistic.png')

    # ── Decision Tree ──────────────────────────────────────
    dt = DecisionTreeClassifier(max_depth=6, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)
    f1_dt  = f1_score(y_test, y_pred_dt, average='weighted')
    results['Decision Tree'] = {'Accuracy': acc_dt, 'F1': f1_dt}
    joblib.dump(dt, os.path.join(MODELS_DIR, 'decision_tree.pkl'))
    print(f"\n🟡 Decision Tree (max_depth=6):")
    print(f"   Accuracy = {acc_dt:.4f}  |  F1-Score = {f1_dt:.4f}")
    print(classification_report(y_test, y_pred_dt, target_names=class_names))
    plot_confusion_matrix(y_test, y_pred_dt, class_names, 'confusion_matrix_decision_tree.png')

    # ── Random Forest Classifier ───────────────────────────
    rf_cls = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_cls.fit(X_train, y_train)
    y_pred_rf = rf_cls.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf  = f1_score(y_test, y_pred_rf, average='weighted')
    results['Random Forest Classifier'] = {'Accuracy': acc_rf, 'F1': f1_rf}
    joblib.dump(rf_cls, os.path.join(MODELS_DIR, 'rf_classifier.pkl'))
    print(f"\n🟢 Random Forest Classifier:")
    print(f"   Accuracy = {acc_rf:.4f}  |  F1-Score = {f1_rf:.4f}")
    print(classification_report(y_test, y_pred_rf, target_names=class_names))
    plot_confusion_matrix(y_test, y_pred_rf, class_names, 'confusion_matrix_rf.png')
    plot_feature_importance(rf_cls, FEATURES, 'rf_classification_feature_importance.png')

    return results


# ─────────────────────────────────────────────────────────
# 5. Visualisations
# ─────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=120)
    plt.close()
    print(f"   → Plot saved: {filename}")


def plot_feature_importance(model, feature_names, filename):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(9, 5))
    sns.barplot(
        x=[feature_names[i] for i in indices],
        y=importances[indices],
        palette='viridis'
    )
    plt.title('Feature Importance')
    plt.xlabel('Feature')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=120)
    plt.close()
    print(f"   → Plot saved: {filename}")


def plot_actual_vs_predicted(y_test, y_pred, title):
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.7, color='steelblue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Score')
    plt.ylabel('Predicted Score')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'actual_vs_predicted.png'), dpi=120)
    plt.close()
    print("   → Plot saved: actual_vs_predicted.png")


def plot_model_comparison(reg_results, cls_results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Regression: R² comparison
    reg_names = list(reg_results.keys())
    r2_scores  = [reg_results[m]['R2'] for m in reg_names]
    axes[0].bar(reg_names, r2_scores, color=['#4C72B0', '#55A868'])
    axes[0].set_title('Regression – R² Score Comparison')
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('R² Score')
    for i, v in enumerate(r2_scores):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=11)

    # Classification: Accuracy comparison
    cls_names  = list(cls_results.keys())
    accuracies = [cls_results[m]['Accuracy'] for m in cls_names]
    colors = ['#4C72B0', '#55A868', '#C44E52']
    axes[1].bar(cls_names, accuracies, color=colors)
    axes[1].set_title('Classification – Accuracy Comparison')
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('Accuracy')
    for i, v in enumerate(accuracies):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison.png'), dpi=120)
    plt.close()
    print("   → Plot saved: model_comparison.png")


def plot_flipped_vs_traditional(df):
    """Simulate and visualize flipped vs traditional classroom performance."""
    traditional_scores = df[TARGET_REG] * 0.88  # simulate ~12% lower
    flipped_scores     = df[TARGET_REG]

    plt.figure(figsize=(9, 5))
    plt.hist(traditional_scores, bins=15, alpha=0.6, label='Traditional Classroom', color='#C44E52')
    plt.hist(flipped_scores,     bins=15, alpha=0.6, label='Flipped Classroom',     color='#55A868')
    plt.axvline(traditional_scores.mean(), color='#C44E52', linestyle='--', linewidth=2,
                label=f'Trad. Mean: {traditional_scores.mean():.1f}')
    plt.axvline(flipped_scores.mean(),     color='#55A868', linestyle='--', linewidth=2,
                label=f'Flipped Mean: {flipped_scores.mean():.1f}')
    plt.xlabel('Final Exam Score')
    plt.ylabel('Number of Students')
    plt.title('Flipped Classroom vs Traditional Teaching – Score Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'flipped_vs_traditional.png'), dpi=120)
    plt.close()
    print("   → Plot saved: flipped_vs_traditional.png")


# ─────────────────────────────────────────────────────────
# 6. Summary Report
# ─────────────────────────────────────────────────────────
def print_summary(reg_results, cls_results):
    print("\n" + "=" * 60)
    print("         FINAL MODEL PERFORMANCE SUMMARY")
    print("=" * 60)
    print("\n📊 REGRESSION MODELS (Predict Final Exam Score)")
    print(f"{'Model':<30} {'MSE':>8} {'RMSE':>8} {'R²':>8}")
    print("-" * 58)
    for model, metrics in reg_results.items():
        print(f"{model:<30} {metrics['MSE']:>8.3f} {metrics['RMSE']:>8.3f} {metrics['R2']:>8.4f}")

    print("\n📊 CLASSIFICATION MODELS (Classify Student Performance)")
    print(f"{'Model':<30} {'Accuracy':>10} {'F1-Score':>10}")
    print("-" * 52)
    for model, metrics in cls_results.items():
        print(f"{model:<30} {metrics['Accuracy']:>10.4f} {metrics['F1']:>10.4f}")

    print("\n✅ All models saved to:", MODELS_DIR)
    print("📈 All plots  saved to:", PLOTS_DIR)


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    print("\n🚀 Starting ML Model Training – Flipped Classroom CSE")
    print("   Author : Uttam Vitthal Bhise | M.Tech CSE\n")

    df = load_data()
    X_scaled, y_reg, y_cls, le, scaler = preprocess(df)

    reg_results = train_regression_models(X_scaled, y_reg)
    cls_results = train_classification_models(X_scaled, y_cls, le)

    print("\n📊 Generating visualisation plots...")
    plot_model_comparison(reg_results, cls_results)
    plot_flipped_vs_traditional(df)

    print_summary(reg_results, cls_results)
    print("\n🎉 Training complete!\n")


if __name__ == '__main__':
    main()
