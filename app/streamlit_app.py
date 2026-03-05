import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.reviewer import review_code, SUPPORTED_LANGUAGES
from agent.scorer import extract_score, get_certification

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QA Spark CodeAgent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/QA%20Spark-CodeAgent-blue?style=for-the-badge")
    st.markdown("## Settings")

    language = st.selectbox(
        "Language / Dialect",
        options=SUPPORTED_LANGUAGES,
        format_func=lambda x: {
            "pyspark": "PySpark (Python) ★",
            "sparksql": "SparkSQL ★",
            "scala": "Scala (Spark) ★",
            "impala": "Impala SQL",
            "python": "Python",
        }.get(x, x.upper()),
    )

    spark_version = st.text_input(
        "Spark Version (Spark languages)",
        value="3.4",
        help="Override Spark version for PySpark, SparkSQL, and Scala reviews. Default: 3.4",
    )
    import os
    os.environ["SPARK_VERSION"] = spark_version

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "QA Spark CodeAgent reviews code for the **Cloudera CDP platform** "
        "and issues a certification when code scores **95+/100**.\n\n"
        "★ = Spark 3.4 optimized"
    )
    st.markdown(
        "**Checks:** best practices · resource efficiency · performance optimization · security"
    )

# ── Main area ───────────────────────────────────────────────────────────────────
st.title("⚡ QA Spark CodeAgent")
st.markdown(
    "Paste your code below and click **Run Review**. "
    "Code must score **95/100 or higher** to receive a Cloudera execution certificate."
)

col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("### Submit Code")

    upload_tab, paste_tab = st.tabs(["Upload File", "Paste Code"])

    with upload_tab:
        uploaded = st.file_uploader(
            "Upload a .sql or .py file",
            type=["sql", "py"],
            label_visibility="collapsed",
        )
        if uploaded:
            code_from_file = uploaded.read().decode("utf-8")
        else:
            code_from_file = None

    with paste_tab:
        lang_placeholder = {
            "impala": "-- Paste your Impala SQL here",
            "sparksql": "-- Paste your SparkSQL query here (spark.sql / spark-sql shell)",
            "pyspark": "# Paste your PySpark code here",
            "scala": "// Paste your Scala Spark code here",
            "python": "# Paste your Python code here",
        }.get(language, "# Paste your code here")

        pasted_code = st.text_area(
            "Code",
            height=400,
            placeholder=lang_placeholder,
            label_visibility="collapsed",
        )

    # Resolve which code to use
    code_to_review = code_from_file or pasted_code or ""

    if code_to_review:
        st.caption(f"{len(code_to_review.splitlines())} lines · {len(code_to_review)} characters")

    run_btn = st.button(
        "▶ Run Review",
        type="primary",
        disabled=not bool(code_to_review.strip()),
        use_container_width=True,
    )

with col_result:
    st.markdown("### Review Result")

    if run_btn and code_to_review.strip():
        with st.spinner(f"Reviewing {language.upper()} code..."):
            try:
                result = review_code(code_to_review, language)
                score = extract_score(result["review"])
                cert = get_certification(score)

                # ── Score banner ─────────────────────────────────────
                if cert["certified"]:
                    st.success(
                        f"✅ **CERTIFIED** — Score: **{score}/100**\n\n"
                        f"{cert['message']}"
                    )
                else:
                    st.error(
                        f"❌ **NOT CERTIFIED** — Score: **{score}/100**\n\n"
                        f"{cert['message']}"
                    )

                # ── Score gauge ──────────────────────────────────────
                st.progress(score / 100)

                # ── Full review ──────────────────────────────────────
                st.markdown("---")
                st.markdown(result["review"])

                # ── Token usage ──────────────────────────────────────
                with st.expander("Token Usage"):
                    usage = result["usage"]
                    st.markdown(
                        f"- **Model:** {result['model']}\n"
                        f"- **Prompt tokens:** {usage['prompt_tokens']:,}\n"
                        f"- **Completion tokens:** {usage['completion_tokens']:,}\n"
                        f"- **Total tokens:** {usage['total_tokens']:,}"
                    )

            except ValueError as e:
                st.error(f"Configuration error: {e}")
            except Exception as e:
                st.error(f"Review failed: {e}")

    elif not run_btn:
        st.info("Enter your code on the left and click **Run Review** to start.")
