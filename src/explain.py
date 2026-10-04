"""Interpretability: which metabolites / pathways drive a 'high risk' call."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def global_importance(model, X, y, n_repeats=5, seed=0) -> pd.DataFrame:
    """Permutation importance measured as drop in PR-AUC (robust for imbalance)."""
    r = permutation_importance(model, X, y, scoring="average_precision", n_repeats=n_repeats,
                               random_state=seed, n_jobs=-1)
    return (pd.DataFrame({"feature": X.columns, "importance": r.importances_mean, "std": r.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def shap_values(model, X):  # pragma: no cover - optional dependency
    """TreeSHAP if `shap` is installed (pip install shap)."""
    import shap
    return shap.TreeExplainer(model).shap_values(X)


def patient_abnormalities(z_row: pd.Series, top=5, cutoff=3.0) -> pd.DataFrame:
    """Top abnormal analytes/ratios for one patient (|robust z| >= cutoff)."""
    s = z_row[[c for c in z_row.index if c.startswith(("z_", "r_"))]]
    s = s[s.abs() >= cutoff].sort_values(key=np.abs, ascending=False).head(top)
    return pd.DataFrame({"analyte": [c[2:] for c in s.index],
                         "type": ["metabolite" if c.startswith("z_") else "ratio" for c in s.index],
                         "robust_z": s.values.round(1),
                         "direction": np.where(s.values > 0, "high", "low")})
