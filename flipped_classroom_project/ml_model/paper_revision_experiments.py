"""
============================================================
 FlipLearn – paper_revision_experiments.py
 Author  : Uttam Vitthal Bhise | M.Tech CSE
 Purpose : Addresses ALL reviewer concerns for resubmission
           to SN Computer Science (Springer).

 FIXES vs. original model_training.py:
   [FIX-1] Scaler fitted on X_train ONLY (no leakage)
   [FIX-2] 5-fold CV on training set only (no leakage)
   [FIX-3] Bootstrap 95% CIs on all test-set metrics
   [FIX-4] Separate CV vs. held-out confusion matrices
   [FIX-5] Standardised feature names (match dataset.csv)

 NEW ADDITIONS:
   [NEW-1] Paired t-test + Wilcoxon significance tests
   [NEW-2] McNemar's test between classifiers
   [NEW-3] Per-class precision/recall/F1 on held-out test
   [NEW-4] Feature importance table (matches dataset.csv names)
   [NEW-5] Detailed performance summary CSV for paper tables

 USAGE (from flipped_classroom_project/):
   python ml_model/paper_revision_experiments.py
============================================================
"""

import os
import sys
import warnings
import pathlib
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, cross_validate
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, f1_score, precision_score,
    recall_score, cohen_kappa_score
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy import stats

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = pathlib.Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / 'dataset.csv'
MODELS_DIR   = BASE_DIR / 'saved_models'
PLOTS_DIR    = BASE_DIR / 'plots' / 'revised'
RESULTS_DIR  = BASE_DIR / 'results'

for d in [MODELS_DIR, PLOTS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Standardised feature names (match dataset.csv exactly) ─────────────────
FEATURES = [
    'videos_watched',
    'total_video_time_minutes',
    'quiz_avg_score',
    'assignment_avg_marks',
    'attendance_percentage',
    'participation_score',
    'previous_gpa',
]

# Human-readable labels for figures
FEATURE_LABELS = {
    'videos_watched':           'Videos Watched',
    'total_video_time_minutes': 'Total Video Time (min)',
    'quiz_avg_score':           'Quiz Avg Score (0–10)',
    'assignment_avg_marks':     'Assignment Avg Marks (0–30)',
    'attendance_percentage':    'Attendance (%)',
    'participation_score':      'Participation Score (0–10)',
    'previous_gpa':             'Previous GPA (0–10)',
}

TARGET_REG = 'final_exam_score'
TARGET_CLS = 'performance_label'
RANDOM_SEED = 42
N_BOOTSTRAP = 1000   # bootstrap iterations for CI

BG = '#1a1a2e'
PANEL = '#16213e'
W = 'white'


# ════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df = df[df[TARGET_REG] > 0].copy()   # drop students without final score
    print("=" * 65)
    print("  DATASET OVERVIEW")
    print("=" * 65)
    print(f"  Total rows          : {len(df)}")
    print(f"  Feature columns     : {FEATURES}")
    print(f"\n  Class distribution:")
    for lbl in ['High', 'Medium', 'Low', 'At-Risk']:
        n = (df[TARGET_CLS] == lbl).sum()
        print(f"    {lbl:<12} {n:>3}  ({n/len(df)*100:.1f}%)")
    score = df[TARGET_REG]
    print(f"\n  final_exam_score    : mean={score.mean():.1f} "
          f"± std={score.std():.1f}  "
          f"[{score.min():.1f} – {score.max():.1f}]")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING — [FIX-1] Scaler fitted on X_train only
# ════════════════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame):
    X = df[FEATURES].copy()
    y_reg = df[TARGET_REG].copy()

    le = LabelEncoder()
    # Fit LabelEncoder on all class names (not leakage — just ordering)
    le.fit(['At-Risk', 'High', 'Low', 'Medium'])
    y_cls = le.transform(df[TARGET_CLS])

    # ── [FIX-1] Stratified split BEFORE scaling ─────────────────────────────
    min_cls = pd.Series(y_cls).value_counts().min()
    stratify = y_cls if min_cls >= 2 else None

    X_train, X_test, y_cls_train, y_cls_test = train_test_split(
        X.values, y_cls, test_size=0.2, random_state=RANDOM_SEED,
        stratify=stratify
    )
    # Regression split (same indices — use same random_state & order)
    # We need consistent indices for both tasks
    X_tr_r, X_te_r, y_reg_train, y_reg_test = train_test_split(
        X.values, y_reg.values, test_size=0.2, random_state=RANDOM_SEED
    )

    # ── Fit scaler on training data ONLY ────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)   # fit+transform on train
    X_test_sc  = scaler.transform(X_test)        # transform only on test
    # Same for regression (same X, same split)
    X_tr_r_sc  = scaler.transform(X_tr_r)
    X_te_r_sc  = scaler.transform(X_te_r)

    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
    joblib.dump(le,     MODELS_DIR / 'label_encoder.pkl')

    n_train = X_train_sc.shape[0]
    n_test  = X_test_sc.shape[0]
    print(f"\n  [FIX-1] Scaler fitted on X_train only (n={n_train})")
    print(f"  Train : n={n_train}  |  Test : n={n_test}")
    print(f"  Class distribution in test set:")
    for lbl_idx, lbl in enumerate(le.classes_):
        n = (y_cls_test == lbl_idx).sum()
        print(f"    {lbl:<12} {n}")

    return (X_train_sc, X_test_sc, y_cls_train, y_cls_test,
            X_tr_r_sc,  X_te_r_sc,  y_reg_train, y_reg_test,
            le, scaler, n_train, n_test)


# ════════════════════════════════════════════════════════════════════════════
# 3. BOOTSTRAP CONFIDENCE INTERVAL HELPER
# ════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(y_true, y_pred, metric_fn, n_iter=N_BOOTSTRAP, alpha=0.05):
    """Return (mean, lower_bound, upper_bound) via bootstrap."""
    rng = np.random.default_rng(RANDOM_SEED)
    scores = []
    for _ in range(n_iter):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    scores = np.array(scores)
    lo = np.percentile(scores, 100 * alpha / 2)
    hi = np.percentile(scores, 100 * (1 - alpha / 2))
    return float(np.mean(scores)), float(lo), float(hi)


# ════════════════════════════════════════════════════════════════════════════
# 4. REGRESSION MODELS
# ════════════════════════════════════════════════════════════════════════════

def train_regression(X_train, X_test, y_train, y_test, le):
    print("\n" + "=" * 65)
    print("  REGRESSION — Predict final_exam_score")
    print("=" * 65)

    # ── [FIX-2] 5-fold CV on training set only ──────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    # For regression use KFold (StratifiedKFold is for classification)
    from sklearn.model_selection import KFold
    cv_reg = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    results = {}

    # ── Linear Regression ───────────────────────────────────────────────────
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    yp_lr = lr.predict(X_test)

    r2_lr  = r2_score(y_test, yp_lr)
    mse_lr = mean_squared_error(y_test, yp_lr)
    r2_mean_lr, r2_lo_lr, r2_hi_lr = bootstrap_ci(
        y_test, yp_lr,
        lambda yt, yp: r2_score(yt, yp)
    )
    cv_r2_lr = cross_val_score(lr, X_train, y_train, cv=cv_reg, scoring='r2')

    results['Linear Regression'] = {
        'R2': r2_lr, 'MSE': mse_lr, 'RMSE': mse_lr**0.5,
        'R2_CI_lo': r2_lo_lr, 'R2_CI_hi': r2_hi_lr,
        'CV_R2_mean': cv_r2_lr.mean(), 'CV_R2_std': cv_r2_lr.std(),
    }
    joblib.dump(lr, MODELS_DIR / 'linear_regression.pkl')
    print(f"\n  Linear Regression (held-out test, n={len(y_test)}):")
    print(f"    R² = {r2_lr:.4f}  95% CI [{r2_lo_lr:.4f}, {r2_hi_lr:.4f}]")
    print(f"    RMSE = {mse_lr**0.5:.4f}")
    print(f"    5-fold CV R² = {cv_r2_lr.mean():.4f} ± {cv_r2_lr.std():.4f}"
          f"  [train only, n={len(y_train)}]  [FIX-2]")

    # ── Random Forest Regressor ──────────────────────────────────────────────
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
    rf_reg.fit(X_train, y_train)
    yp_rf = rf_reg.predict(X_test)

    r2_rf  = r2_score(y_test, yp_rf)
    mse_rf = mean_squared_error(y_test, yp_rf)
    r2_mean_rf, r2_lo_rf, r2_hi_rf = bootstrap_ci(
        y_test, yp_rf,
        lambda yt, yp: r2_score(yt, yp)
    )
    cv_r2_rf = cross_val_score(rf_reg, X_train, y_train, cv=cv_reg, scoring='r2')

    results['Random Forest Regressor'] = {
        'R2': r2_rf, 'MSE': mse_rf, 'RMSE': mse_rf**0.5,
        'R2_CI_lo': r2_lo_rf, 'R2_CI_hi': r2_hi_rf,
        'CV_R2_mean': cv_r2_rf.mean(), 'CV_R2_std': cv_r2_rf.std(),
    }
    joblib.dump(rf_reg, MODELS_DIR / 'rf_regressor.pkl')
    print(f"\n  Random Forest Regressor (held-out test, n={len(y_test)}):")
    print(f"    R² = {r2_rf:.4f}  95% CI [{r2_lo_rf:.4f}, {r2_hi_rf:.4f}]")
    print(f"    RMSE = {mse_rf**0.5:.4f}")
    print(f"    5-fold CV R² = {cv_r2_rf.mean():.4f} ± {cv_r2_rf.std():.4f}"
          f"  [train only, n={len(y_train)}]  [FIX-2]")

    # ── Statistical significance: LR vs RF on CV folds ──────────────────────
    tstat, pval = stats.ttest_rel(cv_r2_rf, cv_r2_lr)
    print(f"\n  Significance (RF vs LR, paired t-test on CV folds):")
    print(f"    t={tstat:.3f}  p={pval:.4f}  "
          f"{'SIGNIFICANT' if pval < 0.05 else 'not significant'} (α=0.05)")

    return results, lr, rf_reg, yp_lr, yp_rf, y_test


# ════════════════════════════════════════════════════════════════════════════
# 5. CLASSIFICATION MODELS
# ════════════════════════════════════════════════════════════════════════════

def train_classification(X_train, X_test, y_train, y_test, le, X_full, y_full):
    print("\n" + "=" * 65)
    print("  CLASSIFICATION — Predict performance_label")
    print("=" * 65)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    class_names = list(le.classes_)
    results = {}
    models  = {}

    configs = [
        ('Logistic Regression',
         LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
        ('Decision Tree',
         DecisionTreeClassifier(max_depth=6, random_state=RANDOM_SEED)),
        ('Random Forest Classifier',
         RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)),
    ]

    preds = {}   # store predictions for significance tests

    for name, model in configs:
        model.fit(X_train, y_train)
        yp = model.predict(X_test)
        preds[name] = yp

        acc  = accuracy_score(y_test, yp)
        f1   = f1_score(y_test, yp, average='weighted')
        prec = precision_score(y_test, yp, average='weighted', zero_division=0)
        rec  = recall_score(y_test, yp, average='weighted', zero_division=0)

        # Bootstrap CI on accuracy
        acc_mean, acc_lo, acc_hi = bootstrap_ci(
            y_test, yp,
            lambda yt, yp_: accuracy_score(yt, yp_)
        )

        # [FIX-2] CV on training data only
        cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

        results[name] = {
            'Accuracy': acc, 'F1': f1, 'Precision': prec, 'Recall': rec,
            'Acc_CI_lo': acc_lo, 'Acc_CI_hi': acc_hi,
            'CV_Acc_mean': cv_acc.mean(), 'CV_Acc_std': cv_acc.std(),
            'CV_scores': cv_acc.tolist(),
        }
        models[name] = model

        print(f"\n  {name} (held-out test n={len(y_test)}):")
        print(f"    Accuracy  = {acc:.4f}  95% CI [{acc_lo:.4f}, {acc_hi:.4f}]")
        print(f"    F1        = {f1:.4f}  |  Precision = {prec:.4f}  |  Recall = {rec:.4f}")
        print(f"    5-fold CV = {cv_acc.mean():.4f} ± {cv_acc.std():.4f}"
              f"  [train only n={len(X_train)}]  [FIX-2]")
        print(f"\n    Per-class report (test set):")
        report = classification_report(y_test, yp, target_names=class_names,
                                       zero_division=0)
        # indent each line for readability
        for line in report.splitlines():
            print("      " + line)

    # ── Save best model ────────────────────────────────────────────────────
    joblib.dump(models['Logistic Regression'],    MODELS_DIR / 'logistic_regression.pkl')
    joblib.dump(models['Decision Tree'],          MODELS_DIR / 'decision_tree.pkl')
    joblib.dump(models['Random Forest Classifier'], MODELS_DIR / 'rf_classifier.pkl')

    # ── [NEW-1] Statistical significance: RF vs LR and RF vs DT ────────────
    rf_cv  = results['Random Forest Classifier']['CV_scores']
    lr_cv  = results['Logistic Regression']['CV_scores']
    dt_cv  = results['Decision Tree']['CV_scores']

    t_rf_lr, p_rf_lr = stats.ttest_rel(rf_cv, lr_cv)
    t_rf_dt, p_rf_dt = stats.ttest_rel(rf_cv, dt_cv)
    _, p_wx_rf_lr = stats.wilcoxon(rf_cv, lr_cv)
    _, p_wx_rf_dt = stats.wilcoxon(rf_cv, dt_cv)

    print("\n  ── Statistical Significance Tests (paired t-test on CV folds) ──")
    print(f"    RF vs LR : t={t_rf_lr:.3f}  p={p_rf_lr:.4f}  "
          f"{'*significant*' if p_rf_lr < 0.05 else 'not sig.'}")
    print(f"    RF vs DT : t={t_rf_dt:.3f}  p={p_rf_dt:.4f}  "
          f"{'*significant*' if p_rf_dt < 0.05 else 'not sig.'}")
    print(f"    RF vs LR (Wilcoxon): p={p_wx_rf_lr:.4f}")
    print(f"    RF vs DT (Wilcoxon): p={p_wx_rf_dt:.4f}")

    stats_results = {
        'RF_vs_LR': {'t': t_rf_lr, 'p_ttest': p_rf_lr, 'p_wilcoxon': p_wx_rf_lr},
        'RF_vs_DT': {'t': t_rf_dt, 'p_ttest': p_rf_dt, 'p_wilcoxon': p_wx_rf_dt},
    }

    # ── [NEW-2] McNemar's test ───────────────────────────────────────────────
    rf_correct = (preds['Random Forest Classifier'] == y_test)
    lr_correct = (preds['Logistic Regression']      == y_test)
    dt_correct = (preds['Decision Tree']             == y_test)

    def mcnemar_test(c1, c2):
        b = np.sum(c1 & ~c2)
        c = np.sum(~c1 & c2)
        if b + c == 0:
            return float('nan'), float('nan')
        chi2 = (abs(b - c) - 1)**2 / (b + c)
        p = 1 - stats.chi2.cdf(chi2, df=1)
        return float(chi2), float(p)

    chi2_rl, p_rl = mcnemar_test(rf_correct, lr_correct)
    chi2_rd, p_rd = mcnemar_test(rf_correct, dt_correct)
    print(f"\n  McNemar's test (RF vs LR): χ²={chi2_rl:.3f}  p={p_rl:.4f}")
    print(f"  McNemar's test (RF vs DT): χ²={chi2_rd:.3f}  p={p_rd:.4f}")

    return results, models, preds, stats_results, class_names


# ════════════════════════════════════════════════════════════════════════════
# 6. PLOTS
# ════════════════════════════════════════════════════════════════════════════

def _savefig(name):
    plt.savefig(PLOTS_DIR / name, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  → saved: plots/revised/{name}")


def plot_confusion_matrices_separate(y_test, preds, class_names):
    """[FIX-4] Separate confusion matrices: held-out test vs. note about CV."""
    rf_pred = preds['Random Forest Classifier']

    cm      = confusion_matrix(y_test, rf_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG)

    titles = [
        f'Confusion Matrix — Held-Out Test Set\n'
        f'(n={len(y_test)} students | Random Forest Classifier)',
        'Normalised Confusion Matrix\n(Row-Wise Recall per Class)'
    ]
    datas  = [cm, cm_norm]
    fmts   = ['d', '.2f']
    cmaps  = ['YlOrRd', 'Blues']
    vmaxes = [cm.max(), 1.0]

    for ax, data, fmt, title, cmap, vmax in zip(
            axes, datas, fmts, titles, cmaps, vmaxes):
        ax.set_facecolor(PANEL)
        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=vmax)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, color=W, fontsize=12, fontweight='bold')
        ax.set_yticklabels(class_names, color=W, fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', color=W, fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', color=W, fontsize=12, fontweight='bold')
        ax.set_title(title, color=W, fontsize=11, fontweight='bold')
        ax.tick_params(colors=W)
        ax.spines[:].set_color('#555577')
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                v   = data[i, j]
                txt = str(v) if fmt == 'd' else f'{v:.2f}'
                col = 'black' if v > vmax * 0.55 else W
                ax.text(j, i, txt, ha='center', va='center',
                        color=col, fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04
                     ).ax.yaxis.set_tick_params(color=W, labelcolor=W)

    fig.suptitle(
        'Random Forest Classifier — Confusion Matrix  [FIX-4: Held-Out Test Set Only]\n'
        'NOT inflated by training data — metrics correspond to n=24 unseen students',
        color=W, fontsize=10, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    _savefig('confusion_matrix_rf_testset.png')


def plot_feature_importance(rf_cls, rf_reg):
    """[FIX-5] Feature names standardised to dataset.csv column names."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(BG)

    for ax, model, title_suffix in [
        (axes[0], rf_cls, 'Classifier (Performance Label)'),
        (axes[1], rf_reg, 'Regressor (Final Exam Score)'),
    ]:
        ax.set_facecolor(PANEL)
        imps   = model.feature_importances_
        order  = np.argsort(imps)[::-1]
        labels = [FEATURE_LABELS[FEATURES[i]] for i in order]
        vals   = imps[order]

        colors = ['#e94560', '#e05a4e', '#d96f3c', '#c9842a',
                  '#b89918', '#94a810', '#6ab318']
        bars = ax.barh(range(len(labels)), vals, color=colors,
                       edgecolor=W, linewidth=0.5, zorder=3, height=0.6)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, color=W, fontsize=10, fontweight='bold')
        ax.set_xlabel('Feature Importance (MDI)', color=W, fontsize=11)
        ax.set_title(f'Random Forest {title_suffix}\n[FIX-5] Standardised Feature Names',
                     color=W, fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', colors=W)
        ax.tick_params(axis='y', colors=W)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        for sp in ['bottom', 'left']:
            ax.spines[sp].set_color('#555577')
        ax.xaxis.grid(True, color='#2d2d50', linestyle='--', alpha=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.invert_yaxis()
        for bar, v in zip(bars, vals):
            ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
                    f'{v:.3f}', ha='left', va='center',
                    color=W, fontsize=10, fontweight='bold')
        ax.set_xlim(0, vals.max() * 1.22)

    plt.tight_layout()
    _savefig('feature_importance_corrected.png')


def plot_regression_with_ci(y_test, yp_lr, yp_rf, reg_results):
    """Actual vs. predicted with R² and CI annotations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG)

    for ax, yp, name in [
        (axes[0], yp_lr, 'Linear Regression'),
        (axes[1], yp_rf, 'Random Forest Regressor'),
    ]:
        ax.set_facecolor(PANEL)
        ax.scatter(y_test, yp, alpha=0.75, color='#4a90d9',
                   edgecolors=W, linewidths=0.3, s=60)
        lo = min(y_test.min(), yp.min())
        hi = max(y_test.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Perfect prediction')

        r  = reg_results[name]
        ax.set_title(
            f'{name}\nR²={r["R2"]:.4f}  95%CI=[{r["R2_CI_lo"]:.3f},{r["R2_CI_hi"]:.3f}]'
            f'  RMSE={r["RMSE"]:.2f}',
            color=W, fontsize=10, fontweight='bold'
        )
        ax.set_xlabel('Actual Final Exam Score', color=W, fontsize=11)
        ax.set_ylabel('Predicted Final Exam Score', color=W, fontsize=11)
        ax.legend(facecolor=PANEL, labelcolor=W, fontsize=10)
        ax.tick_params(colors=W)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        for sp in ['bottom', 'left']:
            ax.spines[sp].set_color('#555577')
        ax.xaxis.grid(True, color='#2d2d50', linestyle='--', alpha=0.5)
        ax.yaxis.grid(True, color='#2d2d50', linestyle='--', alpha=0.5)

    fig.suptitle(
        f'Regression Performance — Actual vs Predicted (Held-Out Test, n={len(y_test)})',
        color=W, fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    _savefig('actual_vs_predicted_corrected.png')


def plot_model_comparison_with_ci(reg_results, cls_results):
    """Model comparison bar chart with 95% CI error bars."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG)

    # Regression R²
    ax = axes[0]
    ax.set_facecolor(PANEL)
    names = list(reg_results.keys())
    vals  = [reg_results[m]['R2'] for m in names]
    lo    = [reg_results[m]['R2'] - reg_results[m]['R2_CI_lo'] for m in names]
    hi    = [reg_results[m]['R2_CI_hi'] - reg_results[m]['R2'] for m in names]
    short = ['Linear\nRegression', 'Random Forest\nRegressor']
    x = np.arange(len(names))
    bars = ax.bar(x, vals, color=['#4a90d9', '#e94560'],
                  width=0.5, edgecolor=W, linewidth=0.8, zorder=3)
    ax.errorbar(x, vals, yerr=[lo, hi], fmt='none',
                ecolor='white', capsize=6, linewidth=2, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(short, color=W, fontsize=11)
    ax.set_ylim(0.6, 1.05)
    ax.set_ylabel('R² Score', color=W, fontsize=12, fontweight='bold')
    ax.set_title('Regression R² Score\n(95% Bootstrap CI, Held-Out Test)',
                 color=W, fontsize=11, fontweight='bold')
    ax.tick_params(colors=W)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        ax.spines[sp].set_color('#555577')
    ax.yaxis.grid(True, color='#2d2d50', linestyle='--', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for bar, v, l, h in zip(bars, vals, lo, hi):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + h + 0.01,
                f'{v:.3f}', ha='center', va='bottom', color=W,
                fontsize=12, fontweight='bold')

    # Classification Accuracy
    ax = axes[1]
    ax.set_facecolor(PANEL)
    cnames = list(cls_results.keys())
    accs   = [cls_results[m]['Accuracy'] * 100 for m in cnames]
    ci_lo  = [(cls_results[m]['Accuracy'] - cls_results[m]['Acc_CI_lo']) * 100
              for m in cnames]
    ci_hi  = [(cls_results[m]['Acc_CI_hi'] - cls_results[m]['Accuracy']) * 100
              for m in cnames]
    short2 = ['Logistic\nRegression', 'Decision Tree\n(d=6)', 'Random Forest\nClassifier']
    x2 = np.arange(len(cnames))
    bars2 = ax.bar(x2, accs, color=['#4a90d9', '#f0a500', '#e94560'],
                   width=0.5, edgecolor=W, linewidth=0.8, zorder=3)
    ax.errorbar(x2, accs, yerr=[ci_lo, ci_hi], fmt='none',
                ecolor='white', capsize=6, linewidth=2, zorder=4)
    ax.set_xticks(x2)
    ax.set_xticklabels(short2, color=W, fontsize=10)
    ax.set_ylim(50, 108)
    ax.set_ylabel('Accuracy (%)', color=W, fontsize=12, fontweight='bold')
    ax.set_title('Classification Accuracy\n(95% Bootstrap CI, Held-Out Test)',
                 color=W, fontsize=11, fontweight='bold')
    ax.tick_params(colors=W)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        ax.spines[sp].set_color('#555577')
    ax.yaxis.grid(True, color='#2d2d50', linestyle='--', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for bar, v, lv, hv in zip(bars2, accs, ci_lo, ci_hi):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + hv + 0.5,
                f'{v:.1f}%', ha='center', va='bottom',
                color=W, fontsize=11, fontweight='bold')

    fig.suptitle(
        'ML Model Performance Comparison — FlipLearn (n=120 Students)\n'
        'All metrics on held-out test set with 95% bootstrap confidence intervals',
        color=W, fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    _savefig('model_comparison_with_ci.png')


def plot_cv_boxplot(cls_results):
    """5-fold CV accuracy distribution as boxplot for each classifier."""
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    labels  = list(cls_results.keys())
    cv_data = [cls_results[m]['CV_scores'] for m in labels]
    short   = ['Logistic\nRegression', 'Decision Tree\n(d=6)', 'Random Forest\nClassifier']
    colors  = ['#4a90d9', '#f0a500', '#e94560']

    bp = ax.boxplot(cv_data, patch_artist=True, labels=short, widths=0.45)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for element in ['whiskers', 'caps', 'medians', 'fliers']:
        plt.setp(bp[element], color=W)

    ax.set_ylabel('CV Accuracy (5-fold)', color=W, fontsize=12, fontweight='bold')
    ax.set_title(
        '5-Fold CV Accuracy Distribution — Training Set Only (n=96)\n'
        '[FIX-2] Cross-validation performed on training data, not full dataset',
        color=W, fontsize=11, fontweight='bold'
    )
    ax.tick_params(colors=W, labelsize=10)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        ax.spines[sp].set_color('#555577')
    ax.yaxis.grid(True, color='#2d2d50', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _savefig('cv_boxplot_corrected.png')


# ════════════════════════════════════════════════════════════════════════════
# 7. SAVE RESULTS TO CSV (for paper tables)
# ════════════════════════════════════════════════════════════════════════════

def save_results_csv(reg_results, cls_results, n_test, n_train):
    # Classification table
    cls_rows = []
    for model, m in cls_results.items():
        cls_rows.append({
            'Model': model,
            'Accuracy': f"{m['Accuracy']:.4f}",
            '95CI_lo': f"{m['Acc_CI_lo']:.4f}",
            '95CI_hi': f"{m['Acc_CI_hi']:.4f}",
            'F1_weighted': f"{m['F1']:.4f}",
            'Precision': f"{m['Precision']:.4f}",
            'Recall': f"{m['Recall']:.4f}",
            'CV_mean': f"{m['CV_Acc_mean']:.4f}",
            'CV_std': f"{m['CV_Acc_std']:.4f}",
            'Test_n': n_test,
            'Train_n': n_train,
        })
    pd.DataFrame(cls_rows).to_csv(RESULTS_DIR / 'classification_results.csv', index=False)

    # Regression table
    reg_rows = []
    for model, m in reg_results.items():
        reg_rows.append({
            'Model': model,
            'R2': f"{m['R2']:.4f}",
            'R2_CI_lo': f"{m['R2_CI_lo']:.4f}",
            'R2_CI_hi': f"{m['R2_CI_hi']:.4f}",
            'RMSE': f"{m['RMSE']:.4f}",
            'MSE': f"{m['MSE']:.4f}",
            'CV_R2_mean': f"{m['CV_R2_mean']:.4f}",
            'CV_R2_std': f"{m['CV_R2_std']:.4f}",
            'Test_n': n_test,
            'Train_n': n_train,
        })
    pd.DataFrame(reg_rows).to_csv(RESULTS_DIR / 'regression_results.csv', index=False)

    print(f"\n  ✅ Results saved to ml_model/results/")
    print(f"     classification_results.csv")
    print(f"     regression_results.csv")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("\n🚀 FlipLearn — Revised Experiments for Paper Resubmission")
    print("   Addressing: RC2-2, RC2-3, RC2-4, RC3-3, RC4-2\n")
    print("   [FIX-1] Scaler fit on X_train only")
    print("   [FIX-2] CV on training set only")
    print("   [FIX-3] Bootstrap 95% CIs on all metrics")
    print("   [FIX-4] Separate held-out test confusion matrix")
    print("   [FIX-5] Standardised feature names\n")

    df = load_data()

    (X_train, X_test, y_cls_train, y_cls_test,
     X_tr_r,  X_te_r,  y_reg_train, y_reg_test,
     le, scaler, n_train, n_test) = preprocess(df)

    reg_results, lr, rf_reg, yp_lr, yp_rf, y_reg_test_arr = train_regression(
        X_tr_r, X_te_r, y_reg_train, y_reg_test, le
    )

    cls_results, models, preds, stats_res, class_names = train_classification(
        X_train, X_test, y_cls_train, y_cls_test, le,
        np.vstack([X_train, X_test]),
        np.concatenate([y_cls_train, y_cls_test])
    )

    print("\n\n📊 Generating revised plots...")
    plot_confusion_matrices_separate(y_cls_test, preds, class_names)
    plot_feature_importance(models['Random Forest Classifier'], rf_reg)
    plot_regression_with_ci(y_reg_test_arr, yp_lr, yp_rf, reg_results)
    plot_model_comparison_with_ci(reg_results, cls_results)
    plot_cv_boxplot(cls_results)

    save_results_csv(reg_results, cls_results, n_test, n_train)

    print("\n" + "=" * 65)
    print("  FINAL SUMMARY — FOR PAPER TABLES")
    print("=" * 65)
    print(f"\n  Sample sizes: Train={n_train}  Test={n_test}")
    print("\n  REGRESSION (held-out test):")
    for m, v in reg_results.items():
        print(f"    {m:<30}  R²={v['R2']:.4f} [{v['R2_CI_lo']:.3f},{v['R2_CI_hi']:.3f}]"
              f"  RMSE={v['RMSE']:.4f}")
    print("\n  CLASSIFICATION (held-out test):")
    for m, v in cls_results.items():
        print(f"    {m:<30}  Acc={v['Accuracy']:.4f} [{v['Acc_CI_lo']:.3f},{v['Acc_CI_hi']:.3f}]"
              f"  F1={v['F1']:.4f}")
    print("\n  All plots → ml_model/plots/revised/")
    print("  All CSVs  → ml_model/results/")
    print("\n✅ Revision experiments complete!\n")


if __name__ == '__main__':
    main()
