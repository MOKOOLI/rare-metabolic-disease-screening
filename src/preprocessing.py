"""
Preprocessing: annotation -> imputation -> batch correction -> PQN -> clinical features.

Mirrors the bench workflow of a clinical biochemist:
  1. identify metabolites by accurate mass (ppm) + retention time
  2. clean the signal (missing values, batch drift, sample dilution)
  3. compare each analyte to the healthy reference population
  4. reason in ratios and pathways, not single numbers
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import METABOLITES, METABOLITE_KEYS, PATHWAYS, RATIOS, PPM_TOL, RT_TOL_MIN, MET_BY_KEY


# ----------------------------------------------------------------- annotation
def ppm_error(observed_mz, theoretical_mz):
    return (observed_mz - theoretical_mz) / theoretical_mz * 1e6


def annotate_features(fmeta: pd.DataFrame, ppm_tol=PPM_TOL, rt_tol=RT_TOL_MIN) -> pd.DataFrame:
    """Match each target metabolite to the best feature by m/z (ppm) and RT.

    Isobaric compounds (Leu/Ile vs allo-Ile, m/z 132.1019) are resolved by RT,
    exactly as on a real chromatographic method.
    Returns one row per metabolite: key, feature_id, ppm_error, rt_error, matched.
    """
    rows = []
    for m in METABOLITES:
        ppm = ppm_error(fmeta["mz"].values, m.mz)
        drt = fmeta["rt"].values - m.rt
        ok = (np.abs(ppm) <= ppm_tol) & (np.abs(drt) <= rt_tol)
        if ok.any():
            score = (ppm[ok] / ppm_tol) ** 2 + (drt[ok] / rt_tol) ** 2
            i = np.flatnonzero(ok)[np.argmin(score)]
            rows.append({"key": m.key, "name": m.name, "theoretical_mz": m.mz,
                         "feature_id": fmeta["feature_id"].iat[i], "observed_mz": fmeta["mz"].iat[i],
                         "ppm_error": ppm[i], "rt_error": drt[i], "matched": True})
        else:
            rows.append({"key": m.key, "name": m.name, "theoretical_mz": m.mz, "feature_id": None,
                         "observed_mz": np.nan, "ppm_error": np.nan, "rt_error": np.nan, "matched": False})
    return pd.DataFrame(rows)


def extract_annotated(table: pd.DataFrame, annotation: pd.DataFrame) -> pd.DataFrame:
    """Return a samples x metabolites intensity matrix (named columns)."""
    ann = annotation[annotation["matched"]]
    out = table[ann["feature_id"].tolist()].copy()
    out.columns = ann["key"].tolist()
    for k in METABOLITE_KEYS:
        if k not in out:
            out[k] = np.nan
    return out[METABOLITE_KEYS]


# ----------------------------------------------------------------- signal cleaning
def impute_half_min(X: pd.DataFrame) -> pd.DataFrame:
    """For values missing because they are below the limit of detection (MNAR)."""
    return X.apply(lambda c: c.fillna(c.min(skipna=True) / 2 if c.notna().any() else 0.0))


def impute_median(X: pd.DataFrame) -> pd.DataFrame:
    """For sporadic peak-picking failures (MCAR).

    Lesson learned in this project: half-minimum imputation of MCAR gaps creates
    artificial extreme LOW values (robust z < -5) that the model reads as disease,
    producing false positives. Choose the imputation to match the missingness
    mechanism.
    """
    return X.fillna(X.median())


def batch_median_correction(X: pd.DataFrame, batch: pd.Series) -> pd.DataFrame:
    """Scale each batch so its per-feature median equals the global median.

    Positives are rare (<3%), so the batch median approximates a pooled QC.
    With real data, prefer pooled-QC based correction (e.g. QC-RLSC).
    """
    X = X.copy()
    global_med = X.median()
    for b in pd.unique(batch):
        idx = (batch == b).values
        X.loc[idx] = X.loc[idx] * (global_med / X.loc[idx].median())
    return X


def pqn_normalize(X: pd.DataFrame, reference: pd.Series | None = None):
    """Probabilistic quotient normalisation (Dieterle et al., 2006).

    Removes sample dilution / punch-volume effects. The median quotient is robust
    to the handful of analytes that a disease pushes up 10-100x.
    """
    if reference is None:
        reference = X.median()
    quotients = X.div(reference, axis=1)
    dilution = quotients.median(axis=1)
    return X.div(dilution, axis=0), reference


# ----------------------------------------------------------------- clinical features
class ClinicalFeatureBuilder:
    """Fit on TRAINING controls only (no leakage), then transform any sample.

    Outputs, per sample:
      z_<met>         robust z-score of log abundance vs healthy reference
      r_<ratio>       robust z-score of log diagnostic ratio
      pw_mean_<path>  signed mean z of metabolites in a pathway
      pw_max_<path>   max |z| in the pathway (an "alarm" for that pathway)
    """

    def fit(self, logX: pd.DataFrame, is_control: np.ndarray):
        ref = logX[is_control]
        self.med_ = ref.median()
        self.mad_ = (ref - self.med_).abs().median() * 1.4826 + 1e-6
        R = self._ratios(logX)[is_control]
        self.rmed_ = R.median()
        self.rmad_ = (R - self.rmed_).abs().median() * 1.4826 + 1e-6
        return self

    @staticmethod
    def _ratios(logX):
        return pd.DataFrame({k: logX[a] - logX[b] for k, (a, b) in RATIOS.items()}, index=logX.index)

    def transform(self, logX: pd.DataFrame) -> pd.DataFrame:
        Z = (logX - self.med_) / self.mad_
        R = (self._ratios(logX) - self.rmed_) / self.rmad_
        feats = {f"z_{k}": Z[k] for k in METABOLITE_KEYS}
        feats.update({f"r_{k}": R[k] for k in RATIOS})
        for p in PATHWAYS:
            keys = [m.key for m in METABOLITES if m.pathway == p]
            feats[f"pw_mean_{p}"] = Z[keys].mean(axis=1)
            feats[f"pw_max_{p}"] = Z[keys].abs().max(axis=1)
        return pd.DataFrame(feats, index=logX.index).clip(-50, 50)


def preprocess_table(table: pd.DataFrame, fmeta: pd.DataFrame, annotation=None, pqn_reference=None,
                     imputation="median"):
    """Raw feature table -> log2 PQN-normalised metabolite matrix."""
    if annotation is None:
        annotation = annotate_features(fmeta)
    X = extract_annotated(table, annotation)
    X = impute_median(X) if imputation == "median" else impute_half_min(X)
    if "batch" in table and table["batch"].nunique() > 1:
        X = batch_median_correction(X, table["batch"])
    X, ref = pqn_normalize(X, pqn_reference)
    logX = np.log2(X.clip(lower=1e-9))
    logX.index = table["sample_id"].values
    return logX, annotation, ref


# ----------------------------------------------------------------- real raw data (optional)
def mzml_to_feature_table(mzml_files, mass_trace_ppm=5.0):  # pragma: no cover - needs pyOpenMS
    """Untargeted feature detection on raw mzML with pyOpenMS (FeatureFinderMetabo).

    Produces the same (table, fmeta) shape as the simulator so the rest of the
    pipeline is unchanged. Install: `pip install pyopenms`.
    """
    import pyopenms as oms

    feature_maps = []
    for f in mzml_files:
        exp = oms.MSExperiment()
        oms.MzMLFile().load(str(f), exp)
        exp.sortSpectra(True)
        mtd = oms.MassTraceDetection(); p = mtd.getDefaults(); p.setValue("mass_error_ppm", mass_trace_ppm); mtd.setParameters(p)
        traces = []; mtd.run(exp, traces, 0)
        epd = oms.ElutionPeakDetection(); split = []; epd.detectPeaks(traces, split)
        ffm = oms.FeatureFindingMetabo(); fm = oms.FeatureMap(); chroms = []
        ffm.run(split, fm, chroms)
        fm.setUniqueIds(); fm.setPrimaryMSRunPath([str(f).encode()])
        feature_maps.append(fm)

    # align + link features across samples
    aligner = oms.MapAlignmentAlgorithmPoseClustering()
    ref_idx = int(np.argmax([fm.size() for fm in feature_maps]))
    aligner.setReference(feature_maps[ref_idx])
    for i, fm in enumerate(feature_maps):
        if i != ref_idx:
            trafo = oms.TransformationDescription(); aligner.align(fm, trafo)
            oms.MapAlignmentTransformer().transformRetentionTimes(fm, trafo, True)
    grouper = oms.FeatureGroupingAlgorithmKD(); consensus = oms.ConsensusMap()
    headers = consensus.getColumnHeaders()
    for i, fm in enumerate(feature_maps):
        h = oms.ColumnHeader(); h.filename = str(mzml_files[i]); h.size = fm.size(); h.unique_id = fm.getUniqueId()
        headers[i] = h
    consensus.setColumnHeaders(headers)
    grouper.group(feature_maps, consensus)

    rows, fmeta = [], []
    for j, cf in enumerate(consensus):
        fid = f"FT{j + 1:04d}"
        fmeta.append({"feature_id": fid, "mz": cf.getMZ(), "rt": cf.getRT() / 60.0})
        vals = {h.getMapIndex(): h.getIntensity() for h in cf.getFeatureList()}
        rows.append([vals.get(i, np.nan) for i in range(len(mzml_files))])
    table = pd.DataFrame(np.array(rows).T, columns=[f["feature_id"] for f in fmeta])
    table.insert(0, "batch", 1)
    table.insert(0, "sample_id", [str(f) for f in mzml_files])
    return table, pd.DataFrame(fmeta)
