import numpy as np

from src.config import MET_BY_KEY
from src.simulate import simulate_concentrations, to_feature_table
from src.preprocessing import annotate_features, preprocess_table, ClinicalFeatureBuilder, ppm_error, pqn_normalize
from src.augmentation import pca_jitter
from src.model import threshold_for_sensitivity


def _small():
    conc, meta = simulate_concentrations(n_controls=400, n_per_disorder=5, seed=1)
    table, fmeta = to_feature_table(conc, seed=1)
    return conc, meta, table, fmeta


def test_exact_masses():
    assert abs(MET_BY_KEY["Phe"].mz - 166.0863) < 1e-3
    assert abs(MET_BY_KEY["C8"].mz - 288.2169) < 1e-3


def test_annotation_resolves_isobars_by_rt():
    _, _, _, fmeta = _small()
    ann = annotate_features(fmeta).set_index("key")
    assert ann.matched.all()
    assert ann.loc["Xle", "feature_id"] != ann.loc["aIle", "feature_id"]
    assert (ann.ppm_error.abs() <= 5).all()


def test_pqn_removes_dilution():
    _, _, table, fmeta = _small()
    logX, _, _ = preprocess_table(table, fmeta)
    X = 2 ** logX
    Xd = X.copy(); Xd.iloc[0] *= 3.0
    ref = X.median()
    a, _ = pqn_normalize(X, ref); b, _ = pqn_normalize(Xd, ref)
    assert np.allclose(a.iloc[0], b.iloc[0], rtol=1e-6)


def test_features_no_nan_and_markers_high():
    _, meta, table, fmeta = _small()
    logX, _, _ = preprocess_table(table, fmeta)
    F = ClinicalFeatureBuilder().fit(logX, (meta.label == 0).values).transform(logX)
    assert not F.isna().any().any()
    pku = (meta.disorder == "PKU").values
    assert F.loc[pku, "r_Phe/Tyr"].median() > 5


def test_augmentation_only_adds_positives():
    _, meta, table, fmeta = _small()
    logX, _, _ = preprocess_table(table, fmeta)
    F = ClinicalFeatureBuilder().fit(logX, (meta.label == 0).values).transform(logX)
    y = meta.label.values
    Xa, ya = pca_jitter(F.reset_index(drop=True), y, meta.disorder.values, n_new_per_pos=3)
    assert (ya[len(y):] == 1).all() and len(ya) == len(y) + 3 * y.sum()


def test_threshold_hits_target_sensitivity():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(1000), np.ones(100)].astype(int)
    s = np.r_[rng.normal(0, 1, 1000), rng.normal(3, 1, 100)]
    thr = threshold_for_sensitivity(y, s, 0.95)
    assert (s[y == 1] >= thr).mean() >= 0.95
