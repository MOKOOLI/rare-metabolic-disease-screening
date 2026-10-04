"""Publication-style figures."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

from .config import PATHWAYS, DISORDERS, METABOLITES
from .augmentation import pca_jitter

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
PAL = ["#2E5EAA", "#E4572E", "#17BEBB", "#FFC914", "#76B041", "#8E44AD", "#555555", "#D35400"]


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def annotation_plot(ann, path):
    a = ann[ann.matched]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.scatter(a.ppm_error, a.rt_error * 60, c=PAL[0], s=40)
    for _, r in a.iterrows():
        ax.annotate(r.key, (r.ppm_error, r.rt_error * 60), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axvspan(-5, 5, color=PAL[2], alpha=0.08, label="+/-5 ppm window")
    ax.axhline(0, c="grey", lw=.5); ax.axvline(0, c="grey", lw=.5)
    ax.set_xlabel("mass error (ppm)"); ax.set_ylabel("RT error (s)")
    ax.set_title(f"Metabolite annotation: {len(a)}/{len(ann)} targets matched by accurate mass + RT")
    ax.legend(frameon=False); _save(fig, path)


def pathway_heatmap(X, disorder, path):
    cols = [f"pw_mean_{p}" for p in PATHWAYS]
    df = X[cols].copy(); df["d"] = disorder
    M = df.groupby("d").mean().reindex(["none", *DISORDERS])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    im = ax.imshow(M.values, cmap="RdBu_r", vmin=-8, vmax=8, aspect="auto")
    ax.set_xticks(range(len(cols)), [c.replace("pw_mean_", "").replace("_", " ") for c in cols], rotation=40, ha="right")
    ax.set_yticks(range(len(M)), ["healthy" if i == "none" else i for i in M.index])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if abs(v) > 2:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if abs(v) > 5 else "black")
    fig.colorbar(im, ax=ax, label="mean pathway robust z")
    ax.set_title("Pathway-level signatures: each disorder lights up its own pathway"); _save(fig, path)


def roc_pr(y, scores: dict, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for i, (n, s) in enumerate(scores.items()):
        f, t, _ = roc_curve(y, s); p, r, _ = precision_recall_curve(y, s)
        a1.plot(f, t, c=PAL[i % 8], lw=1.4, label=f"{n}  AUC={roc_auc_score(y, s):.3f}")
        a2.plot(r, p, c=PAL[i % 8], lw=1.4, label=f"{n}  AP={average_precision_score(y, s):.3f}")
    a1.plot([0, 1], [0, 1], "k--", lw=.6); a1.set_xlim(0, .2)
    a1.set(xlabel="False positive rate (zoom 0-0.2)", ylabel="Sensitivity", title="ROC (held-out test)")
    a2.set(xlabel="Recall (sensitivity)", ylabel="Precision (PPV)", title="Precision-Recall (held-out test)")
    a1.legend(fontsize=6.5, frameon=False); a2.legend(fontsize=6.5, frameon=False, loc="lower left"); _save(fig, path)


def importance_bar(imp, path, top=15):
    d = imp.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(d.feature, d.importance, xerr=d["std"], color=[PAL[1] if f.startswith("r_") else PAL[0] if f.startswith("z_")
                                                         else PAL[2] for f in d.feature])
    ax.set_xlabel("drop in PR-AUC when permuted")
    ax.set_title("What drives the model (blue=metabolite, red=ratio, teal=pathway)"); _save(fig, path)


def baseline_vs_ml(base, ml, tier, path):
    keys = ["sensitivity", "specificity", "PPV"]
    fig, ax = plt.subplots(figsize=(7, 3.6)); x = np.arange(len(keys)); bars = []
    for off, d, c, lab in [(-.27, base, PAL[6], f"classic cut-off (FP={base['FP']}, FN={base['FN']})"),
                           (0, ml, PAL[0], f"ML stand-alone (FP={ml['FP']}, FN={ml['FN']})"),
                           (.27, tier, PAL[1], f"two-tier: cut-off -> ML (FP={tier['FP']}, FN={tier['FN']})")]:
        bars += ax.bar(x + off, [d[k] for k in keys], .27, color=c, label=lab)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .01, f"{b.get_height():.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(x, keys); ax.set_ylim(0, 1.25); ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title("Held-out test set"); _save(fig, path)


def subgroup_plot(sub, path):
    s = sub[sub.group.str.startswith(("confounder", "healthy"))]
    fig, ax = plt.subplots(figsize=(6.5, 3.2)); x = np.arange(len(s))
    ax.bar(x - .27, s["flagged_by_cutoff_%"], .27, color=PAL[6], label="cut-off")
    ax.bar(x, s["flagged_by_ML_%"], .27, color=PAL[0], label="ML stand-alone")
    ax.bar(x + .27, s["flagged_by_two_tier_%"], .27, color=PAL[1], label="two-tier")
    ax.set_xticks(x, [g.split(":")[1] for g in s.group], rotation=20)
    ax.set_ylabel("% healthy babies flagged (false positives)")
    ax.set_title("Confounders: where single cut-offs fail and ratios/pathways help"); ax.legend(frameon=False); _save(fig, path)


def augmentation_pca(X, y, g, path, seed=0):
    Xa, ya = pca_jitter(X, y, g, seed=seed)
    p = PCA(2, random_state=seed).fit(X.values)
    P0, Pa = p.transform(X.values), p.transform(Xa.values[len(X):])
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.scatter(*P0[y == 0].T, s=2, c="#BBBBBB", label="controls")
    ax.scatter(*Pa.T, s=6, c=PAL[2], alpha=.35, label="synthetic positives (PCA jitter)")
    ax.scatter(*P0[y == 1].T, s=14, c=PAL[1], label="real positives")
    ax.set(xlabel="PC1", ylabel="PC2", title="Synthetic minority samples stay on the disease manifold")
    ax.legend(frameon=False, fontsize=7, markerscale=2); _save(fig, path)


def confounder_scatter(X, meta, path):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    groups = [("healthy", (meta.confounder == "none") & (meta.label == 0), "#BBBBBB"),
              ("TPN", meta.confounder == "TPN", PAL[3]), ("premature", meta.confounder == "premature", PAL[4]),
              ("PKU", meta.disorder == "PKU", PAL[1])]
    for n, m, c in groups:
        ax.scatter(X.loc[m.values, "z_Phe"], X.loc[m.values, "r_Phe/Tyr"], s=8 if n == "healthy" else 22, c=c, label=n)
    ax.axvline(3, ls="--", c="k", lw=.6); ax.axhline(3, ls="--", c="k", lw=.6)
    ax.set(xlabel="Phe robust z", ylabel="Phe/Tyr ratio robust z",
           title="Why ratios matter: TPN raises Phe, but not Phe/Tyr"); ax.legend(frameon=False); _save(fig, path)
