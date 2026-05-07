import streamlit as st
import json
import os
import tempfile
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent.extractor import ingest_and_extract
from agent.report_generator import generate_report_from_results

st.set_page_config(
    page_title="Model Validation Agent",
    page_icon="shield",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stDecoration"] { background: #ff007f !important; background-image: none !important; }
[data-testid="stHeader"] { background: transparent !important; border-bottom: none !important; }
</style>
""", unsafe_allow_html=True)

st.title("Model Validation Agent")
st.caption("Upload a model documentation PDF and get an automated validation working paper")

with st.sidebar:
    st.header("About")
    st.markdown("""
This agent:
- Reads your model documentation PDF
- Answers 12 standard validation questions
- Cites page-level evidence for each answer
- Generates a structured working paper

**Validation sections:**
- Model Identification
- Model Development
- Model Performance
- Governance
    """)
    st.divider()
    st.caption("Powered by Claude API + FAISS + RAG")

uploaded_file = st.file_uploader(
    "Upload model documentation PDF",
    type=["pdf"],
    help="Upload a model validation document, technical specification, or model development report"
)

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.0f} KB)")

    if st.button("Run Validation Agent", type="primary"):
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.read())

            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir, exist_ok=True)

            with st.status("Running validation agent...", expanded=True) as status:
                st.write("Extracting and chunking PDF...")
                try:
                    results = ingest_and_extract(
                        pdf_path=pdf_path,
                        output_dir=output_dir,
                        api_key=os.getenv("ANTHROPIC_API_KEY")
                    )
                    st.write("Generating report...")
                    report_md = generate_report_from_results(results, output_dir=output_dir)
                    status.update(label="Validation complete", state="complete")
                except Exception as e:
                    status.update(label=f"Error: {e}", state="error")
                    st.error(str(e))
                    st.stop()

            passed = sum(1 for s in results.values() for a in s["answers"] if a["status"] == "passed")
            needs_review = sum(1 for s in results.values() for a in s["answers"] if a["status"] == "needs_review")
            not_found = sum(1 for s in results.values() for a in s["answers"] if a["status"] == "not_found")
            total = passed + needs_review + not_found
            score = round((passed / total) * 100) if total > 0 else 0

            st.divider()
            st.subheader("Results summary")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Compliance score", f"{score}%")
            col2.metric("Passed", passed)
            col3.metric("Needs review", needs_review, delta_color="inverse")
            col4.metric("Not found", not_found, delta_color="inverse")

            st.divider()

            for section_key, section in results.items():
                st.subheader(section["sheet"])
                for a in section["answers"]:
                    if a["status"] == "passed":
                        color = "green"
                        icon = "PASS"
                    elif a["status"] == "needs_review":
                        color = "orange"
                        icon = "REVIEW"
                    else:
                        color = "red"
                        icon = "FAIL"

                    with st.expander(f"[{icon}] {a['id']} — {a['question']}"):
                        st.markdown(f"**Status:** :{color}[{a['status'].replace('_', ' ').title()}]")
                        st.markdown(f"**Answer:** {a.get('answer', 'N/A')}")
                        if a.get("evidence_quote"):
                            st.info(f"Evidence: {a['evidence_quote']}")
                        if a.get("page_reference"):
                            st.caption(f"Page reference: {a['page_reference']}")
                        if a.get("notes"):
                            st.caption(f"Notes: {a['notes']}")

            st.divider()
            st.subheader("Download report")
            st.download_button(
                label="Download validation report (.md)",
                data=report_md,
                file_name="validation_report.md",
                mime="text/markdown"
            )

else:
    st.info("Upload a PDF document above to begin validation")
    st.markdown("""
**Example documents to try:**
- Any bank credit risk model documentation
- A machine learning model technical specification
- The Basel Committee model risk principles document
    """)
