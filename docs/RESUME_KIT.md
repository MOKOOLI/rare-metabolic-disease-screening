# Resume & application kit

## CV bullets (pick 2-3)

**Rare Metabolic Disease Screening AI Pipeline (LC-HR-MS)** · Python, scikit-learn, pyOpenMS, Streamlit · github.com/MOKOOLI/rare-metabolic-disease-screening

- Built an end-to-end pipeline that screens 8 inborn errors of metabolism from LC-HR-MS data, covering accurate-mass annotation (±5 ppm, RT-resolved isobars), PQN normalisation, batch correction and pathway-level clinical features.
- Benchmarked 6 ML models × 3 class-imbalance strategies under 5-fold CV. As a second-tier filter, cut false positives by 40 % vs classic newborn-screening cut-offs (PPV 0.41 → 0.54), and identified the specific disorder with 100 % top-1 accuracy on a held-out set.
- Designed a 12,400-sample simulator encoding disease signatures and real-world confounders (TPN, prematurity, pivalate antibiotics, MCT oil), and showed which false positives ML can fix and which need an analytical (MS/MS) solution.
- Delivered a clinician-facing Streamlit app with a human-in-the-loop review queue and audit trail, plus unit tests and CI.

*(Always add: "on a biochemically informed synthetic cohort". Interviewers respect honesty and will ask.)*

## Motivation letter / SOP paragraph

> During my MSc in Clinical Biochemistry I became interested in why newborn screening, one of the most successful public-health programmes, still recalls many healthy babies. To explore this, I built an open-source pipeline that reasons the way a clinical biochemist does: it identifies metabolites by accurate mass and retention time, compares each analyte to a healthy reference population, and relies on diagnostic ratios and pathway patterns instead of single cut-offs. Machine learning on top of these features reduced simulated false positives by 40 %. The most valuable result, though, was learning where ML cannot help: pivaloylcarnitine from antibiotics is isobaric with isovalerylcarnitine, so separating them needs MS/MS, not a better model. I want to continue at the intersection of analytical biochemistry and data science, validating such methods on real clinical cohorts.

## Freelance profile text (Kaya / Upwork / Freelancer)

**Headline:** Clinical Biochemist | Metabolomics & LC-MS Data Analysis | Python, ML, Data Visualisation

> I'm a clinical biochemistry specialist (MSc) who turns lab and mass-spec data into clear answers. I can help with LC-MS / metabolomics data processing (annotation, normalisation, QC), statistical analysis and ML models for biomedical data, publication-ready figures, and clean, well-documented Excel/Python workflows. Recent project: an open-source AI pipeline for rare metabolic disease screening (GitHub link). Every project comes with documented code and a short report you can actually read.

**Gig ideas that match this project**
1. Metabolomics / LC-MS data analysis (PCA, volcano plots, pathway enrichment)
2. Biomedical statistics for theses and papers (SPSS/R/Python)
3. Lab data cleaning & Excel dashboards
4. Scientific figures and results sections

## Interview questions you should be ready for

- *Why synthetic data?* Few confirmed positives in public data, plus PHI restrictions. The simulator encodes known markers, and the mzML loader is ready for real data.
- *Why not accuracy?* At 3 % prevalence, "always negative" is 97 % accurate. I fix sensitivity and report specificity/PPV.
- *How did you avoid leakage?* Split first. Reference ranges, PQN reference and augmentation are fitted on training data only.
- *Biggest mistake?* Half-minimum imputation of random gaps created fake low values and false positives. Matching imputation to the missingness mechanism fixed it.
- *What would you do with real data?* Retrospective validation, gestational-age covariates, MS/MS for isobars, calibration, silent-mode prospective run.
