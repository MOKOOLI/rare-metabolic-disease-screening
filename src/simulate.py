"""
Synthetic LC-HR-MS cohort generator.

Why synthetic? Public metabolomics repositories (MetaboLights, Metabolomics
Workbench) contain very few confirmed rare-disease positives, and newborn
screening data are protected health information. This simulator encodes known
biochemistry so the full pipeline can be developed, tested and benchmarked
end-to-end. Swap in real data via `src.io.load_feature_table`.

Realism built in on purpose:
  * 8 inborn errors of metabolism with literature-style marker patterns
  * mild / variant phenotypes (hard positives)
  * clinical confounders that fool single-analyte cut-offs:
      - TPN (parenteral nutrition): all amino acids up, ratios normal
      - prematurity: transient tyrosine / phenylalanine elevation
      - heterozygous carriers: mild marker shifts
      - pivalate antibiotics: false C5 elevation (pivaloylcarnitine is isobaric with C5)
      - MCT-oil feeding: C8 and C10 up together, C8/C10 normal
  * analytical artefacts: spot-volume factor, batch effects, m/z ppm error,
    RT drift, isotopologues, unannotated features, missing values
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import METABOLITES, METABOLITE_KEYS, MET_BY_KEY, DISORDERS, DISORDER_KEYS

AA = ["Phe", "Tyr", "Xle", "aIle", "Val", "Met", "Cit", "Arg", "Orn", "Gln"]
ACYL = ["C0", "C2", "C3", "C5", "C5DC", "C8", "C10"]


def simulate_concentrations(n_controls=10000, n_per_disorder=30, mild_fraction=0.20, seed=42):
    """Return (concentrations DataFrame, metadata DataFrame)."""
    rng = np.random.default_rng(seed)
    n_pos = n_per_disorder * len(DISORDER_KEYS)
    n = n_controls + n_pos

    log_med = np.log([m.ref_median for m in METABOLITES])
    log_sd = np.log([m.ref_gsd for m in METABOLITES])

    # correlated biological variation: amino-acid and acylcarnitine latent factors
    aa_factor = rng.normal(0, 0.10, n)
    ac_factor = rng.normal(0, 0.10, n)
    logc = log_med + rng.normal(0, 1, (n, len(METABOLITES))) * log_sd
    for j, k in enumerate(METABOLITE_KEYS):
        logc[:, j] += aa_factor if k in AA else ac_factor
    conc = pd.DataFrame(np.exp(logc), columns=METABOLITE_KEYS)

    meta = pd.DataFrame({"sample_id": [f"S{i:05d}" for i in range(n)],
                         "disorder": "none", "phenotype": "healthy", "confounder": "none"})

    def shift(idx, markers, softness=1.0):
        for k, (lo, hi) in markers.items():
            fold = np.exp(rng.uniform(np.log(lo), np.log(hi), len(idx))) ** softness
            conc.loc[idx, k] *= fold

    # positives
    pos_idx = np.arange(n_controls, n)
    for d_i, d in enumerate(DISORDER_KEYS):
        idx = pos_idx[d_i * n_per_disorder:(d_i + 1) * n_per_disorder]
        mild = rng.random(len(idx)) < mild_fraction
        shift(idx[~mild], DISORDERS[d]["markers"])
        shift(idx[mild], DISORDERS[d]["markers"], softness=0.6)
        meta.loc[idx, "disorder"] = d
        meta.loc[idx, "phenotype"] = np.where(mild, "mild", "classic")

    # confounders among controls
    ctrl = rng.permutation(n_controls)
    groups = {"TPN": 0.03, "premature": 0.05, "carrier": 0.05, "pivalate": 0.01, "MCT_oil": 0.01}
    start = 0
    for g, frac in groups.items():
        idx = ctrl[start:start + int(frac * n_controls)]
        start += len(idx)
        meta.loc[idx, "confounder"] = g
        if g == "TPN":
            f = np.exp(rng.uniform(np.log(1.8), np.log(3.5), len(idx)))
            for k in AA:
                conc.loc[idx, k] *= f * np.exp(rng.normal(0, 0.08, len(idx)))
        elif g == "premature":
            conc.loc[idx, "Tyr"] *= np.exp(rng.uniform(np.log(1.5), np.log(3.5), len(idx)))
            conc.loc[idx, "Phe"] *= np.exp(rng.uniform(np.log(1.2), np.log(1.8), len(idx)))
            conc.loc[idx, "Met"] *= np.exp(rng.uniform(np.log(1.1), np.log(1.6), len(idx)))
        elif g == "carrier":
            for i in idx:
                d = DISORDER_KEYS[rng.integers(len(DISORDER_KEYS))]
                shift(np.array([i]), DISORDERS[d]["markers"], softness=0.25)
        elif g == "pivalate":
            conc.loc[idx, "C5"] *= np.exp(rng.uniform(np.log(4), np.log(15), len(idx)))
        elif g == "MCT_oil":
            f = np.exp(rng.uniform(np.log(3), np.log(10), len(idx)))
            conc.loc[idx, "C8"] *= f
            conc.loc[idx, "C10"] *= f * np.exp(rng.normal(0, 0.1, len(idx)))

    conc.insert(0, "sample_id", meta["sample_id"])
    meta["label"] = (meta["disorder"] != "none").astype(int)
    shuffle = rng.permutation(n)
    return conc.iloc[shuffle].reset_index(drop=True), meta.iloc[shuffle].reset_index(drop=True)


def to_feature_table(conc: pd.DataFrame, n_batches=5, n_unknown=25, missing_rate=0.01, seed=7):
    """Convert concentrations into an untargeted-style LC-HR-MS feature table.

    Returns (intensity table [samples x features], feature metadata [feature_id, mz, rt]).
    Feature ids are anonymous (FT0001...) so annotation must be done from m/z + RT.
    """
    rng = np.random.default_rng(seed)
    n = len(conc)
    batch = rng.integers(1, n_batches + 1, n)
    spot = np.exp(rng.normal(0, 0.18, n))            # DBS punch volume / hematocrit

    cols, feats = {}, []
    fid = 0

    def add(mz, rt, intensity):
        nonlocal fid
        fid += 1
        name = f"FT{fid:04d}"
        feats.append({"feature_id": name, "mz": round(mz, 5), "rt": round(rt, 3)})
        cols[name] = intensity

    for m in METABOLITES:
        response = 10 ** rng.uniform(4, 6)
        batch_eff = np.exp(rng.normal(0, 0.12, n_batches + 1))[batch]
        noise = np.exp(rng.normal(0, 0.08, n))
        inten = conc[m.key].values * response * spot * batch_eff * noise
        ppm_err = rng.normal(0, 1.3)
        add(m.mz * (1 + ppm_err * 1e-6), m.rt + rng.normal(0, 0.04), inten)
        if m.key in ("Phe", "Tyr", "C0", "Gln"):       # 13C isotopologue (M+1)
            add(m.mz + 1.00335, m.rt + rng.normal(0, 0.02), inten * 0.1 * np.exp(rng.normal(0, 0.05, n)))

    for _ in range(n_unknown):                           # unannotated background features
        add(rng.uniform(100, 450), rng.uniform(1, 12),
            10 ** rng.uniform(3.5, 5.5) * spot * np.exp(rng.normal(0, 0.4, n)))

    table = pd.DataFrame(cols)
    mask = rng.random(table.shape) < missing_rate
    table = table.mask(mask)
    order = rng.permutation(table.columns)               # anonymise column order
    table = table[order]
    table.insert(0, "batch", batch)
    table.insert(0, "sample_id", conc["sample_id"].values)
    fmeta = pd.DataFrame(feats).set_index("feature_id").loc[order].reset_index()
    return table, fmeta
