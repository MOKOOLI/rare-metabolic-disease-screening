"""
Interactive screening demo with a human-in-the-loop review queue.

    streamlit run app/streamlit_app.py
"""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.screening import load_bundle, screen  # noqa: E402

st.set_page_config(page_title="Rare IEM Screening AI", page_icon="🧬", layout="wide")
st.title("🧬 Rare metabolic disease screening (LC-HR-MS)")
st.caption("Research prototype on synthetic data. NOT a medical device. Every HIGH RISK call must be "
           "reviewed by a clinical biochemist on the raw spectra before any clinical action.")


@st.cache_resource
def bundle():
    return load_bundle()


B = bundle()
fmeta = pd.read_csv(ROOT / "data/raw/feature_metadata.csv")

with st.sidebar:
    st.header("Model")
    st.write("**Two-tier screening**")
    st.write("Tier 1: classic cut-off (any primary marker > 99.5th centile of healthy controls)")
    st.write(f"Tier 2: **{B['tier2_name']}** re-ranks tier-1 positives (threshold `{B['tier2_threshold']:.3f}`)")
    st.write(f"Stand-alone risk score: {B['model_name']}")
    up = st.file_uploader("Upload LC-HR-MS feature table (CSV)", type="csv")
    st.download_button("Download example table", (ROOT / "data/raw/example_feature_table.csv").read_bytes(),
                       "example_feature_table.csv")

table = pd.read_csv(up) if up else pd.read_csv(ROOT / "data/raw/example_feature_table.csv")
res = screen(table, fmeta, B)

c1, c2, c3 = st.columns(3)
c1.metric("Samples screened", len(res))
c2.metric("High risk", int(res.call.str.startswith("HIGH").sum()))
c3.metric("Low risk", int((res.call == "low risk").sum()))

st.subheader("Results")
st.dataframe(res.style.apply(lambda r: ["background-color:#fde2e1" if r.call.startswith("HIGH") else ""] * len(r), axis=1),
             use_container_width=True)

st.subheader("👩‍🔬 Specialist review queue (human-in-the-loop)")
queue = res[res.call.str.startswith("HIGH")]
if "decisions" not in st.session_state:
    st.session_state.decisions = []
for _, r in queue.iterrows():
    with st.expander(f"{r.sample_id}  |  risk {r.risk_score:.2f}  |  suspected: {r.suspected_disorder}"):
        st.write("**Key abnormalities:**", r.key_abnormalities or "none above |z|>=3")
        st.write("Check: peak shape, isotope pattern, MS/MS match, RT, internal standard recovery.")
        col_a, col_b, col_c = st.columns(3)
        for col, label in [(col_a, "Confirm: refer for diagnostic test"), (col_b, "Reject: analytical artefact"),
                           (col_c, "Request repeat sample")]:
            if col.button(label, key=f"{r.sample_id}-{label}"):
                st.session_state.decisions.append({"sample_id": r.sample_id, "decision": label,
                                                   "time": datetime.now().isoformat(timespec="seconds")})
if st.session_state.decisions:
    log = pd.DataFrame(st.session_state.decisions)
    st.write("Audit trail"); st.dataframe(log)
    st.download_button("Export audit log", log.to_csv(index=False), "review_audit_log.csv")
