"""
PCA-space jitter: synthetic rare-positive samples.

For each disorder separately (to keep its biochemical signature):
  1. project positives into a PCA space fitted on the training set
  2. add low-amplitude Gaussian noise, scaled per component by the spread of
     that disorder's positives
  3. map back to feature space

Applied INSIDE each training fold only. Augmenting before the split leaks
near-copies of test patients into training and inflates sensitivity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def pca_jitter(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_new_per_pos=5,
               alpha=0.15, n_components=0.99, seed=0):
    rng = np.random.default_rng(seed)
    pca = PCA(n_components=n_components, random_state=seed).fit(X.values)
    new_X, new_y = [], []
    for g in pd.unique(groups[y == 1]):
        idx = np.flatnonzero((y == 1) & (groups == g))
        if len(idx) < 2:
            continue
        P = pca.transform(X.values[idx])
        scale = P.std(axis=0, ddof=1) * alpha + 1e-9
        base = P[rng.integers(len(idx), size=len(idx) * n_new_per_pos)]
        synth = pca.inverse_transform(base + rng.normal(0, 1, base.shape) * scale)
        new_X.append(synth)
        new_y.append(np.ones(len(synth), dtype=int))
    if not new_X:
        return X, y
    Xa = pd.concat([X, pd.DataFrame(np.vstack(new_X), columns=X.columns)], ignore_index=True)
    return Xa, np.concatenate([y, np.concatenate(new_y)])
