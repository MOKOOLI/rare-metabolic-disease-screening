"""
End-to-end run:  simulate -> annotate -> normalise -> features -> models -> evaluate -> explain.

    python run_pipeline.py            # full run (~5-10 min on a laptop)
    python run_pipeline.py --quick    # smaller cohort, for CI / smoke test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.config import PRIMARY_MARKER, DISORDERS
from src.simulate import simulate_concentrations, to_feature_table
from src.preprocessing import preprocess_table, ClinicalFeatureBuilder
from src.model import (get_models, oof_scores, fit_with_strategy, score, threshold_for_sensitivity,
                       screening_metrics, classic_cutoff_baseline)
from src.explain import global_importance
from src import plots

ROOT = Path(__file__).resolve().parent


def main(quick=False, target_sens=0.97, seed=42):
    t0 = time.time()
    for d in ["data/raw", "data/processed", "figures", "results", "models"]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)

    # 1. data -------------------------------------------------------------
    n_ctrl, n_pos = (2000, 15) if quick else (12000, 50)
    conc, meta = simulate_concentrations(n_controls=n_ctrl, n_per_disorder=n_pos, seed=seed)
    table, fmeta = to_feature_table(conc, seed=seed)
    table.to_csv(ROOT / "data/raw/feature_table.csv", index=False)
    fmeta.to_csv(ROOT / "data/raw/feature_metadata.csv", index=False)
    meta.to_csv(ROOT / "data/raw/sample_metadata.csv", index=False)
    table.head(25).to_csv(ROOT / "data/raw/example_feature_table.csv", index=False)
    print(f"[data] {len(table)} samples, {table.shape[1] - 2} LC-HR-MS features, {meta.label.sum()} positives")

    # 2. split before ANY fitting ------------------------------------------
    strat = meta["disorder"].values
    tr_idx, te_idx = train_test_split(np.arange(len(meta)), test_size=0.2, stratify=strat, random_state=seed)
    tab_tr, tab_te = table.iloc[tr_idx].reset_index(drop=True), table.iloc[te_idx].reset_index(drop=True)
    m_tr, m_te = meta.iloc[tr_idx].reset_index(drop=True), meta.iloc[te_idx].reset_index(drop=True)
    y_tr, y_te = m_tr["label"].values, m_te["label"].values
    g_tr = m_tr["disorder"].values

    # 3. preprocessing ----------------------------------------------------
    logX_tr, annotation, pqn_ref = preprocess_table(tab_tr, fmeta)
    logX_te, _, _ = preprocess_table(tab_te, fmeta, annotation=annotation, pqn_reference=pqn_ref)
    annotation.to_csv(ROOT / "results/annotation_report.csv", index=False)
    print(f"[annotate] {annotation.matched.sum()}/{len(annotation)} metabolites matched, "
          f"max |ppm| = {annotation.ppm_error.abs().max():.2f}")

    builder = ClinicalFeatureBuilder().fit(logX_tr, is_control=(y_tr == 0))
    X_tr, X_te = builder.transform(logX_tr), builder.transform(logX_te)
    pd.concat([X_tr.assign(split="train", disorder=g_tr), X_te.assign(split="test", disorder=m_te.disorder.values)]) \
        .to_csv(ROOT / "data/processed/feature_matrix.csv")
    print(f"[features] {X_tr.shape[1]} clinical features (z-scores, ratios, pathway scores)")

    # 4. classic cut-off baseline -------------------------------------------
    base_tr = classic_cutoff_baseline(X_tr[y_tr == 0], X_tr, PRIMARY_MARKER).astype(bool)
    base_te = classic_cutoff_baseline(X_tr[y_tr == 0], X_te, PRIMARY_MARKER)
    baseline = screening_metrics(y_te, base_te, 0.5)

    # 5. model comparison (5-fold OOF on train) -----------------------------
    models = get_models(seed)
    strategies = ["none", "class_weight", "pca_jitter"]
    rows, oof_store = [], {}
    for name, mdl in models.items():
        for st in strategies:
            if quick and st == "class_weight":
                continue
            ts = time.time()
            oof = oof_scores(mdl, X_tr, y_tr, g_tr, st, seed=seed)
            thr = threshold_for_sensitivity(y_tr, oof, target_sens)
            m = screening_metrics(y_tr, oof, thr)
            rows.append({"model": name, "strategy": st, **{k: m[k] for k in
                         ["PR_AUC", "ROC_AUC", "sensitivity", "specificity", "PPV", "FP"]},
                         "seconds": round(time.time() - ts, 1)})
            oof_store[(name, st)] = oof
            # two-tier: ML re-ranks cut-off positives; t2 keeps every tier-1 true positive (OOF)
            t2 = oof[base_tr & (y_tr == 1)].min() - 1e-9
            rows[-1]["tier2_threshold"] = t2
            rows[-1]["tier2_oof_FP"] = int((base_tr & (oof >= t2) & (y_tr == 0)).sum())
            print(f"  {name:22s} {st:13s} PR-AUC={m['PR_AUC']:.3f} spec@{target_sens:.0%}sens={m['specificity']:.4f} "
                  f"tier2 FP={rows[-1]['tier2_oof_FP']} (tier1 FP={int((base_tr & (y_tr == 0)).sum())})")
    cv = pd.DataFrame(rows).sort_values(["specificity", "PR_AUC"], ascending=False)
    cv.to_csv(ROOT / "results/cv_model_comparison.csv", index=False)

    best = cv.iloc[0]
    best_name, best_st = best["model"], best["strategy"]
    thr = threshold_for_sensitivity(y_tr, oof_store[(best_name, best_st)], target_sens)
    final = fit_with_strategy(models[best_name], X_tr, y_tr, g_tr, best_st, seed)
    s_te = score(final, X_te)
    test = screening_metrics(y_te, s_te, thr)
    print(f"[best] {best_name} + {best_st}: test sensitivity={test['sensitivity']:.3f} "
          f"specificity={test['specificity']:.4f} FN={test['FN']} FP={test['FP']}")

    # two-tier system, chosen on OOF only (no test peeking)
    t2row = cv.sort_values(["tier2_oof_FP", "PR_AUC"], ascending=[True, False]).iloc[0]
    t2_name, t2_st, t2 = t2row["model"], t2row["strategy"], float(t2row["tier2_threshold"])
    t2_model = final if (t2_name, t2_st) == (best_name, best_st) else fit_with_strategy(models[t2_name], X_tr, y_tr, g_tr, t2_st, seed)
    s2_te = score(t2_model, X_te)
    tier2_flag = (base_te.astype(bool) & (s2_te >= t2)).astype(float)
    two_tier = screening_metrics(y_te, tier2_flag, 0.5)
    print(f"[two-tier] cut-off -> {t2_name} + {t2_st}: test sensitivity={two_tier['sensitivity']:.3f} "
          f"FP {baseline['FP']} -> {two_tier['FP']}  FN={two_tier['FN']}")

    # all models on test (for ROC/PR figure)
    test_scores = {}
    for name, mdl in models.items():
        st = cv[cv.model == name].iloc[0]["strategy"]
        test_scores[f"{name} ({st})"] = score(fit_with_strategy(mdl, X_tr, y_tr, g_tr, st, seed), X_te)

    # 6. disorder classifier (which IEM?) -----------------------------------
    dclf = RandomForestClassifier(n_estimators=400, random_state=seed, class_weight="balanced")
    dclf.fit(X_tr[y_tr == 1], g_tr[y_tr == 1])
    d_acc = float((dclf.predict(X_te[y_te == 1]) == m_te.disorder.values[y_te == 1]).mean())

    # 7. error analysis by subgroup ----------------------------------------
    pred_ml = s_te >= thr
    sub = []
    for grp, mask in [*[(f"confounder:{c}", (m_te.confounder == c).values) for c in m_te.confounder.unique() if c != "none"],
                      ("healthy:no confounder", ((m_te.confounder == "none") & (y_te == 0)).values),
                      *[(f"disorder:{d}", (m_te.disorder == d).values) for d in DISORDERS],
                      ("phenotype:mild", (m_te.phenotype == "mild").values)]:
        if mask.sum():
            sub.append({"group": grp, "n": int(mask.sum()),
                        "flagged_by_cutoff_%": round(100 * base_te[mask].mean(), 1),
                        "flagged_by_ML_%": round(100 * pred_ml[mask].mean(), 1),
                        "flagged_by_two_tier_%": round(100 * tier2_flag[mask].mean(), 1)})
    sub = pd.DataFrame(sub)
    sub.to_csv(ROOT / "results/subgroup_analysis.csv", index=False)

    # 8. explainability ---------------------------------------------------
    imp = global_importance(final, X_te, y_te, n_repeats=3 if quick else 5, seed=seed)
    imp.to_csv(ROOT / "results/feature_importance.csv", index=False)

    # 9. figures ----------------------------------------------------------
    plots.annotation_plot(annotation, ROOT / "figures/annotation_ppm_rt.png")
    plots.pathway_heatmap(X_te, m_te.disorder.values, ROOT / "figures/pathway_heatmap.png")
    plots.roc_pr(y_te, test_scores, ROOT / "figures/roc_pr_curves.png")
    plots.importance_bar(imp, ROOT / "figures/feature_importance.png")
    plots.baseline_vs_ml(baseline, test, two_tier, ROOT / "figures/baseline_vs_ml.png")
    plots.subgroup_plot(sub, ROOT / "figures/subgroup_flags.png")
    plots.augmentation_pca(X_tr, y_tr, g_tr, ROOT / "figures/pca_jitter_augmentation.png", seed)
    plots.confounder_scatter(X_te, m_te, ROOT / "figures/phe_tyr_confounders.png")

    # 10. persist ---------------------------------------------------------
    joblib.dump({"model": final, "threshold": thr, "builder": builder, "annotation": annotation,
                 "pqn_reference": pqn_ref, "feature_names": list(X_tr.columns), "disorder_model": dclf,
                 "model_name": f"{best_name} + {best_st}", "target_sensitivity": target_sens,
                 "tier2_model": t2_model, "tier2_threshold": t2, "tier2_name": f"{t2_name} + {t2_st}",
                 "control_reference": X_tr[y_tr == 0][[f"z_{k}" for k in PRIMARY_MARKER.values()]]},
                ROOT / "models/screening_bundle.joblib", compress=3)
    metrics = {"data": {"n_samples": int(len(meta)), "n_positive": int(meta.label.sum()),
                        "n_test": int(len(y_te)), "n_test_positive": int(y_te.sum()), "synthetic": True},
               "best_model": f"{best_name} + {best_st}", "target_sensitivity": target_sens,
               "test_ML": test, "test_classic_cutoff": baseline, "test_two_tier": two_tier,
               "two_tier_model": f"{t2_name} + {t2_st}", "disorder_top1_accuracy": d_acc,
               "runtime_s": round(time.time() - t0, 1)}
    (ROOT / "results/metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"[done] {metrics['runtime_s']} s. Disorder identification accuracy = {d_acc:.3f}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--target-sensitivity", type=float, default=0.97)
    a = ap.parse_args()
    main(quick=a.quick, target_sens=a.target_sensitivity)
