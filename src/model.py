"""
Model zoo, imbalance strategies, screening-grade threshold selection and metrics.

Screening philosophy: a missed case (false negative) means irreversible harm,
a false positive means a recall test. So we pick the decision threshold that
reaches a target sensitivity on out-of-fold training predictions, then report
what that costs in specificity / PPV on a held-out test set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .augmentation import pca_jitter


def get_models(seed=0) -> dict:
    models = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000)),
        "RandomForest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed),
        "SVM_RBF": make_pipeline(StandardScaler(), SVC(C=2.0, gamma="scale", random_state=seed)),
        "MLP_deep": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64, 32), alpha=1e-3,
                                                                  early_stopping=True, max_iter=400, random_state=seed)),
    }
    try:  # optional, used automatically when installed
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9,
                                          colsample_bytree=0.8, eval_metric="aucpr", random_state=seed)
    except ImportError:
        pass
    return models


def score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    s = model.decision_function(X)
    return 1 / (1 + np.exp(-s))


def fit_with_strategy(model, X, y, groups, strategy, seed=0):
    m = clone(model)
    if strategy == "pca_jitter":
        Xa, ya = pca_jitter(X, y, groups, seed=seed)
        m.fit(Xa, ya)
    elif strategy == "class_weight":
        w = np.where(y == 1, (y == 0).sum() / max((y == 1).sum(), 1), 1.0)
        try:
            last = m.steps[-1][0] if hasattr(m, "steps") else None
            m.fit(X, y, **({f"{last}__sample_weight": w} if last else {"sample_weight": w}))
        except TypeError:  # e.g. MLP has no sample_weight -> fall back to jitter
            Xa, ya = pca_jitter(X, y, groups, seed=seed)
            m.fit(Xa, ya)
    else:
        m.fit(X, y)
    return m


def oof_scores(model, X, y, groups, strategy, n_splits=5, seed=0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    strat = np.where(y == 1, groups, "none")
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, strat):
        m = fit_with_strategy(model, X.iloc[tr], y[tr], groups[tr], strategy, seed)
        oof[va] = score(m, X.iloc[va])
    return oof


def threshold_for_sensitivity(y, s, target=0.97):
    pos = np.sort(s[y == 1])
    k = int(np.floor((1 - target) * len(pos)))
    return float(pos[k]) - 1e-12


def screening_metrics(y, s, thr):
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
            "FNR": fn / (tp + fn), "FPR": fp / (fp + tn), "PPV": tp / max(tp + fp, 1),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "ROC_AUC": roc_auc_score(y, s), "PR_AUC": average_precision_score(y, s), "threshold": thr}


def classic_cutoff_baseline(Z_train_ctrl: pd.DataFrame, Z: pd.DataFrame, primary_markers: dict, pct=99.5):
    """Classic newborn-screening rule: flag if ANY primary marker > 99.5th centile of controls."""
    flags = np.zeros(len(Z), dtype=bool)
    for d, k in primary_markers.items():
        cut = np.percentile(Z_train_ctrl[f"z_{k}"], pct)
        flags |= Z[f"z_{k}"].values > cut
    return flags.astype(float)
