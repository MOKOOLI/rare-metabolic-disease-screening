"""Inference API used by the Streamlit app and by downstream scripts."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import DISORDERS
from .preprocessing import preprocess_table
from .explain import patient_abnormalities
from .model import score

DEFAULT_BUNDLE = Path(__file__).resolve().parents[1] / "models" / "screening_bundle.joblib"


def load_bundle(path=DEFAULT_BUNDLE):
    return joblib.load(path)


def screen(table: pd.DataFrame, fmeta: pd.DataFrame, bundle) -> pd.DataFrame:
    """Screen a batch of samples. Returns one row per sample with risk + reasons."""
    logX, _, _ = preprocess_table(table, fmeta, annotation=bundle["annotation"],
                                  pqn_reference=bundle["pqn_reference"])
    F = bundle["builder"].transform(logX)[bundle["feature_names"]]
    risk = score(bundle["model"], F)
    tier1 = np.zeros(len(F), dtype=bool)
    for col in bundle["control_reference"].columns:
        tier1 |= F[col].values > np.percentile(bundle["control_reference"][col], 99.5)
    tier2 = tier1 & (score(bundle["tier2_model"], F) >= bundle["tier2_threshold"])
    disorder_proba = bundle["disorder_model"].predict_proba(F)
    classes = bundle["disorder_model"].classes_
    out = []
    for i, sid in enumerate(F.index):
        top = patient_abnormalities(F.iloc[i])
        j = int(np.argmax(disorder_proba[i]))
        high = bool(tier2[i])
        out.append({
            "sample_id": sid,
            "risk_score": float(risk[i]),
            "tier1_cutoff_flag": bool(tier1[i]),
            "call": "HIGH RISK - specialist review" if high else "low risk",
            "suspected_disorder": DISORDERS[classes[j]]["label"] if high else "",
            "disorder_confidence": float(disorder_proba[i, j]) if high else np.nan,
            "key_abnormalities": "; ".join(f"{a} {d} (z={z})" for a, d, z in
                                           zip(top["analyte"], top["direction"], top["robust_z"])),
        })
    return pd.DataFrame(out)
