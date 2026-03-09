import os
import sys
import re
import datetime
from pathlib import Path

import requests
import streamlit as st

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

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .score-box {
        border-radius: 12px;
        padding: 20px 28px;
        text-align: center;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    .score-certified   { background: #d4edda; color: #155724; border: 2px solid #28a745; }
    .score-warning     { background: #fff3cd; color: #856404; border: 2px solid #ffc107; }
    .score-fail        { background: #f8d7da; color: #721c24; border: 2px solid #dc3545; }
    .section-card {
        background: #f8f9fa;
        border-left: 4px solid #0d6efd;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .hist-item {
        font-size: 0.85rem;
        padding: 6px 0;
        border-bottom: 1px solid #eee;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ── Language metadata ──────────────────────────────────────────────────────────
LANG_LABELS = {
    "pyspark":  "PySpark (Python) ★",
    "sparksql": "SparkSQL ★",
    "scala":    "Scala Spark ★",
    "impala":   "Impala SQL",
    "python":   "Python",
}
LANG_EXTS = {".py": None, ".sql": None, ".scala": "scala"}
LANG_PLACEHOLDERS = {
    "pyspark":  "# Paste your PySpark code here",
    "sparksql": "-- Paste your SparkSQL query here",
    "scala":    "// Paste your Scala Spark code here",
    "impala":   "-- Paste your Impala SQL here",
    "python":   "# Paste your Python script here",
}


def detect_language(filename: str, content: str) -> str:
    """Auto-detect language from filename + content."""
    ext = Path(filename).suffix.lower()
    if ext == ".scala":
        return "scala"
    if ext == ".sql":
        if re.search(r"spark\.sql|SparkSession|spark-sql", content, re.I):
            return "sparksql"
        return "impala"
    if ext == ".py":
        if re.search(r"SparkSession|SparkContext|from pyspark|import pyspark", content):
            return "pyspark"
        return "python"
    return "python"


def score_color_class(score: int) -> str:
    if score >= 95:
        return "score-certified"
    if score >= 75:
        return "score-warning"
    return "score-fail"


def score_emoji(score: int) -> str:
    if score >= 95:
        return "✅"
    if score >= 75:
        return "⚠️"
    return "❌"


def fetch_gitlab_file(base_url: str, token: str, project: str, branch: str, filepath: str) -> str:
    """Fetch raw file content from a GitLab instance."""
    encoded_path = filepath.replace("/", "%2F")
    encoded_project = project.replace("/", "%2F")
    url = f"{base_url.rstrip('/')}/api/v4/projects/{encoded_project}/repository/files/{encoded_path}/raw?ref={branch}"
    headers = {"PRIVATE-TOKEN": token}
    resp = requests.get(url, headers=headers, timeout=15)
    if resp.status_code == 404:
        raise ValueError(f"File not found: {filepath} on branch {branch}")
    if resp.status_code == 401:
        raise ValueError("Invalid GitLab token — check your Personal Access Token.")
    resp.raise_for_status()
    return resp.text


def run_review(code: str, language: str, source_label: str):
    """Run the review and update session state."""
    with st.spinner(f"Reviewing {LANG_LABELS.get(language, language)} code with {os.environ.get('AZURE_DEPLOYED_MODEL', 'AI model')}..."):
        result = review_code(code, language)
        score = extract_score(result["review"])
        cert = get_certification(score)

        st.session_state.last_result = {
            "result": result,
            "score": score,
            "cert": cert,
            "source": source_label,
            "language": language,
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "code": code,
        }

        # Add to history (keep last 10)
        st.session_state.history.insert(0, {
            "source": source_label,
            "language": LANG_LABELS.get(language, language),
            "score": score,
            "certified": cert["certified"],
            "timestamp": st.session_state.last_result["timestamp"],
        })
        if len(st.session_state.history) > 10:
            st.session_state.history.pop()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ QA Spark CodeAgent")
    st.caption("Cloudera CDP Code Review & Certification")
    st.divider()

    st.markdown("### Settings")
    spark_version = st.text_input(
        "Spark Version",
        value=os.environ.get("SPARK_VERSION", "3.4"),
        help="Applies to PySpark, SparkSQL, Scala reviews. Default: 3.4",
    )
    os.environ["SPARK_VERSION"] = spark_version

    pass_threshold = st.slider(
        "Certification Threshold",
        min_value=80, max_value=100, value=95, step=1,
        help="Minimum score required to earn certification"
    )
    os.environ["PASS_THRESHOLD"] = str(pass_threshold)

    st.divider()
    st.markdown("### Review History")
    if st.session_state.history:
        for h in st.session_state.history:
            badge = "✅" if h["certified"] else "❌"
            st.markdown(
                f'<div class="hist-item">{badge} <b>{h["score"]}/100</b> — '
                f'{h["language"]}<br><small>{h["source"]} · {h["timestamp"]}</small></div>',
                unsafe_allow_html=True,
            )
        if st.button("Clear History", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()
    else:
        st.caption("No reviews yet.")

    st.divider()
    st.markdown(
        "**★** = Spark 3.4 reviewed \n\n"
        "**Checks:** best practices · performance · resource efficiency · security\n\n"
        "Score **95+** = certified for Cloudera execution"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.title("⚡ QA Spark CodeAgent")
st.markdown(
    "Submit code for AI-powered review against Cloudera CDP best practices. "
    "Code must score **95/100 or higher** to receive a certification badge."
)

left, right = st.columns([1, 1], gap="large")

# ── LEFT: Input ────────────────────────────────────────────────────────────────
with left:
    st.markdown("### Submit Code")

    repo_tab, paste_tab, upload_tab = st.tabs([
        "🔗 Code Repository",
        "📋 Paste Code",
        "📁 Upload File",
    ])

    code_to_review = ""
    detected_language = None
    source_label = ""

    # ── Tab 1: Repository ──────────────────────────────────────────────────────
    with repo_tab:
        st.markdown("Connect to your GitLab repository to pull a file directly.")

        gitlab_url = st.text_input(
            "GitLab Base URL",
            value="https://gitlab.com",
            placeholder="https://gitlab.yourcompany.com",
        )
        gitlab_token = st.text_input(
            "Personal Access Token",
            type="password",
            placeholder="glpat-xxxxxxxxxxxxxxxxxxxx",
            help="GitLab → User Settings → Access Tokens → create with `read_repository` scope",
        )
        repo_project = st.text_input(
            "Project Path",
            placeholder="group/project-name",
        )
        repo_branch = st.text_input("Branch", value="main")
        repo_filepath = st.text_input(
            "File Path in Repo",
            placeholder="src/jobs/my_spark_job.py",
        )

        fetch_btn = st.button("⬇ Fetch File", use_container_width=True)

        if fetch_btn:
            if not all([gitlab_url, gitlab_token, repo_project, repo_branch, repo_filepath]):
                st.error("Please fill in all fields above.")
            else:
                with st.spinner("Fetching from GitLab..."):
                    try:
                        fetched = fetch_gitlab_file(
                            gitlab_url, gitlab_token,
                            repo_project, repo_branch, repo_filepath,
                        )
                        st.session_state["repo_code"] = fetched
                        st.session_state["repo_file"] = repo_filepath
                        st.success(f"Fetched: `{repo_filepath}` ({len(fetched.splitlines())} lines)")
                    except Exception as e:
                        st.error(str(e))

        if "repo_code" in st.session_state:
            code_to_review = st.session_state["repo_code"]
            source_label = st.session_state.get("repo_file", "repo file")
            detected_language = detect_language(source_label, code_to_review)
            with st.expander("Preview fetched code"):
                st.code(code_to_review[:2000] + ("..." if len(code_to_review) > 2000 else ""), language="python")

    # ── Tab 2: Paste ───────────────────────────────────────────────────────────
    with paste_tab:
        paste_lang = st.selectbox(
            "Language",
            options=SUPPORTED_LANGUAGES,
            format_func=lambda x: LANG_LABELS.get(x, x),
            key="paste_lang",
        )
        pasted = st.text_area(
            "Paste your code here",
            height=380,
            placeholder=LANG_PLACEHOLDERS.get(paste_lang, ""),
            label_visibility="collapsed",
            key="paste_code",
        )
        if pasted.strip():
            code_to_review = pasted
            detected_language = paste_lang
            source_label = "pasted code"
            st.caption(f"{len(pasted.splitlines())} lines · {len(pasted):,} chars")

    # ── Tab 3: Upload ──────────────────────────────────────────────────────────
    with upload_tab:
        uploaded = st.file_uploader(
            "Upload a code file",
            type=["py", "sql", "scala"],
            label_visibility="collapsed",
        )
        if uploaded:
            file_content = uploaded.read().decode("utf-8", errors="replace")
            code_to_review = file_content
            source_label = uploaded.name
            detected_language = detect_language(uploaded.name, file_content)
            st.success(f"Loaded: `{uploaded.name}` ({len(file_content.splitlines())} lines)")
            st.caption(f"Auto-detected language: **{LANG_LABELS.get(detected_language, detected_language)}**")
            with st.expander("Preview"):
                st.code(file_content[:2000] + ("..." if len(file_content) > 2000 else ""), language="python")

    st.divider()

    # ── Language override + Run button ─────────────────────────────────────────
    if code_to_review:
        final_language = st.selectbox(
            "Confirm / Override Language",
            options=SUPPORTED_LANGUAGES,
            index=SUPPORTED_LANGUAGES.index(detected_language) if detected_language in SUPPORTED_LANGUAGES else 0,
            format_func=lambda x: LANG_LABELS.get(x, x),
        )

        run_btn = st.button(
            "▶  Run Review",
            type="primary",
            use_container_width=True,
        )

        if run_btn:
            try:
                run_review(code_to_review, final_language, source_label)
            except Exception as e:
                st.error(f"Review failed: {e}")
    else:
        st.info("Use one of the tabs above to load your code, then click **Run Review**.")


# ── RIGHT: Results ─────────────────────────────────────────────────────────────
with right:
    st.markdown("### Review Result")

    data = st.session_state.last_result

    if not data:
        st.markdown("""
        **How it works:**
        1. Load your code using one of the three input methods
        2. Confirm the language and click **Run Review**
        3. The AI reviews your code against Cloudera CDP best practices
        4. You get a score, detailed findings, and fixed code
        5. Score **95+** = certified to run on Cloudera

        **Supported languages:**
        - ⭐ PySpark · SparkSQL · Scala Spark (Spark 3.4)
        - Impala SQL · Python
        """)
    else:
        score = data["score"]
        cert = data["cert"]
        result = data["result"]
        color_class = score_color_class(score)
        emoji = score_emoji(score)

        # ── Score banner ───────────────────────────────────────────────────────
        st.markdown(
            f'<div class="score-box {color_class}">'
            f'{emoji} {score}/100 — {cert["status"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Score bar ──────────────────────────────────────────────────────────
        st.progress(score / 100)

        # ── Meta row ───────────────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        m1.metric("Score", f"{score}/100")
        m2.metric("Threshold", f"{cert['threshold']}/100")
        m3.metric("Language", LANG_LABELS.get(data["language"], data["language"]))
        st.caption(
            f"Source: `{data['source']}` · "
            f"Model: {result['model']} · "
            f"Provider: {result['provider'].upper()} · "
            f"Reviewed at {data['timestamp']}"
        )

        st.divider()

        # ── Sectioned output ───────────────────────────────────────────────────
        review_text = result["review"]

        # Split into named sections
        section_pattern = re.compile(r"#{1,3}\s*\d+\.\s+(.+)", re.MULTILINE)
        section_splits = list(section_pattern.finditer(review_text))

        if section_splits:
            sections = {}
            for i, match in enumerate(section_splits):
                start = match.start()
                end = section_splits[i + 1].start() if i + 1 < len(section_splits) else len(review_text)
                sections[match.group(0).strip()] = review_text[start:end].strip()

            tab_labels = list(sections.keys())
            tabs = st.tabs([re.sub(r"#{1,3}\s*\d+\.\s*", "", t) for t in tab_labels])
            for tab, (label, content) in zip(tabs, sections.items()):
                with tab:
                    st.markdown(content)
        else:
            # Fallback: render as one block
            st.markdown(review_text)

        st.divider()

        # ── Download + copy actions ────────────────────────────────────────────
        dl1, dl2 = st.columns(2)
        with dl1:
            report_md = (
                f"# QA Spark CodeAgent Review\n\n"
                f"**Source:** {data['source']}  \n"
                f"**Language:** {LANG_LABELS.get(data['language'], data['language'])}  \n"
                f"**Score:** {score}/100  \n"
                f"**Status:** {cert['status']}  \n"
                f"**Model:** {result['model']}  \n"
                f"**Reviewed:** {data['timestamp']}\n\n"
                f"---\n\n{review_text}"
            )
            st.download_button(
                "⬇ Download Report (.md)",
                data=report_md,
                file_name=f"qa_review_{data['language']}_{score}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with dl2:
            # Extract optimized code block
            code_match = re.search(r"```(?:\w+)?\n([\s\S]+?)```", review_text)
            if code_match:
                st.download_button(
                    "⬇ Download Optimized Code",
                    data=code_match.group(1),
                    file_name=f"optimized_{data['language']}.{'sql' if data['language'] in ('impala','sparksql') else 'scala' if data['language'] == 'scala' else 'py'}",
                    mime="text/plain",
                    use_container_width=True,
                )

        # ── Token usage ────────────────────────────────────────────────────────
        if result["usage"]["total_tokens"]:
            with st.expander("Token Usage"):
                u = result["usage"]
                st.markdown(
                    f"- **Prompt:** {u['prompt_tokens']:,} tokens\n"
                    f"- **Completion:** {u['completion_tokens']:,} tokens\n"
                    f"- **Total:** {u['total_tokens']:,} tokens"
                )
