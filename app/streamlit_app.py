"""
QA Spark CodeAgent — Streamlit UI (Multi-Agent Edition)

Tabs:
  Input — three sub-tabs: GitLab repo, Paste code, Upload file
  Results — composite score, per-agent breakdown, corrected code, full review text
  Cost & ROI — per-review cost, lifetime ROI summary, cost-by-language chart
"""
import json
import sys
from pathlib import Path

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config
from agent.orchestrator import Orchestrator
from agent.reviewer import review_code, SUPPORTED_LANGUAGES
from agent.scorer import extract_score, get_certification
from cost.roi_logger import roi_summary

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QA Spark CodeAgent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

LANG_DISPLAY = {
    "impala":   "Impala SQL",
    "pyspark":  "PySpark",
    "sparksql": "SparkSQL",
    "scala":    "Scala Spark",
    "python":   "Python",
}

SCORE_CSS = """
<style>
.score-box {
    padding: 20px; border-radius: 12px; text-align: center;
    font-size: 2.8rem; font-weight: bold; margin-bottom: 16px;
}
.score-certified   { background: #d4edda; color: #155724; border: 2px solid #28a745; }
.score-not-certified { background: #f8d7da; color: #721c24; border: 2px solid #dc3545; }
</style>
"""
st.markdown(SCORE_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — history and settings
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar(cfg: Config):
    with st.sidebar:
        st.title("⚡ QA Spark CodeAgent")
        st.caption(f"Model: **{cfg.model_name}** | Mode: **{cfg.agent_mode.upper()}**")
        st.divider()

        st.subheader("Review History")
        history = st.session_state.get("history", [])
        if not history:
            st.caption("No reviews yet this session.")
        for i, entry in enumerate(reversed(history[-10:])):
            score = entry.get("score", "?")
            lang  = entry.get("language", "?").upper()
            cert  = "✅" if entry.get("certified") else "❌"
            label = f"{cert} {lang} — {score}/100"
            if st.button(label, key=f"hist_{i}", use_container_width=True):
                st.session_state["show_result"] = entry

        st.divider()
        st.subheader("Settings")
        mode = st.selectbox(
            "Agent Mode",
            ["multi", "single"],
            index=0 if cfg.agent_mode == "multi" else 1,
            key="agent_mode_select",
        )
        st.session_state["agent_mode_override"] = mode

        spark_ver = st.selectbox(
            "Spark Version",
            ["3.4", "3.5", "3.3"],
            index=0,
            key="spark_version_select",
        )
        st.session_state["spark_version_override"] = spark_ver


# ══════════════════════════════════════════════════════════════════════════════
# Input tab
# ══════════════════════════════════════════════════════════════════════════════
def render_input_tab(cfg: Config):
    st.header("Submit Code for Review")

    col1, col2 = st.columns([2, 1])
    with col1:
        language = st.selectbox(
            "Language / Dialect",
            SUPPORTED_LANGUAGES,
            format_func=lambda x: LANG_DISPLAY.get(x, x),
            key="language_select",
        )
    with col2:
        st.metric("Pass Threshold", f"{cfg.pass_threshold}/100")

    sub_tab_gitlab, sub_tab_paste, sub_tab_upload = st.tabs([
        "🔗 GitLab Repo", "📋 Paste Code", "📁 Upload File"
    ])

    code = None
    file_name = "<pasted>"

    # ── GitLab tab ─────────────────────────────────────────────────────────
    with sub_tab_gitlab:
        st.caption("Fetch a file directly from your GitLab repository.")
        gl_url   = st.text_input("GitLab Project URL", placeholder="https://gitlab.company.com/group/project")
        gl_token = st.text_input("GitLab Access Token", type="password")
        gl_ref   = st.text_input("Branch / Tag / Commit", value="main")
        gl_path  = st.text_input("File Path in Repo", placeholder="src/jobs/my_spark_job.py")

        if st.button("Fetch & Review", key="btn_gitlab", type="primary"):
            if not (gl_url and gl_token and gl_path):
                st.error("Please fill in all GitLab fields.")
            else:
                with st.spinner("Fetching file from GitLab…"):
                    code, file_name = _fetch_gitlab_file(gl_url, gl_token, gl_ref, gl_path)

    # ── Paste tab ───────────────────────────────────────────────────────────
    with sub_tab_paste:
        pasted = st.text_area(
            "Paste your code here",
            height=320,
            placeholder="SELECT * FROM ...",
            key="paste_area",
        )
        if st.button("Review Pasted Code", key="btn_paste", type="primary"):
            if pasted.strip():
                code = pasted
            else:
                st.error("Please paste some code first.")

    # ── Upload tab ──────────────────────────────────────────────────────────
    with sub_tab_upload:
        uploaded = st.file_uploader(
            "Upload a code file",
            type=["sql", "py", "scala", "txt"],
            key="file_uploader",
        )
        if st.button("Review Uploaded File", key="btn_upload", type="primary"):
            if uploaded:
                code = uploaded.read().decode("utf-8", errors="replace")
                file_name = uploaded.name
                _auto_detect_language(uploaded.name)
            else:
                st.error("Please upload a file first.")

    return code, language, file_name


def _fetch_gitlab_file(project_url: str, token: str, ref: str, file_path: str):
    import requests
    import urllib.parse
    project_url = project_url.rstrip("/")
    parts = project_url.split("/")
    if len(parts) < 5:
        st.error("Invalid GitLab URL format.")
        return None, "<unknown>"
    gitlab_base  = "/".join(parts[:3])
    namespace    = parts[3]
    project      = parts[4]
    encoded_path = urllib.parse.quote_plus(f"{namespace}/{project}")
    encoded_file = urllib.parse.quote_plus(file_path)
    api_url = (
        f"{gitlab_base}/api/v4/projects/{encoded_path}"
        f"/repository/files/{encoded_file}/raw?ref={ref}"
    )
    resp = requests.get(api_url, headers={"PRIVATE-TOKEN": token}, timeout=15)
    if resp.status_code == 200:
        return resp.text, file_path.split("/")[-1]
    st.error(f"GitLab API error {resp.status_code}: {resp.text[:200]}")
    return None, "<unknown>"


def _auto_detect_language(filename: str):
    ext = Path(filename).suffix.lower()
    mapping = {".sql": "impala", ".py": "pyspark", ".scala": "scala"}
    detected = mapping.get(ext)
    if detected and detected in SUPPORTED_LANGUAGES:
        st.session_state["language_select"] = detected


# ══════════════════════════════════════════════════════════════════════════════
# Run review
# ══════════════════════════════════════════════════════════════════════════════
def run_review(code: str, language: str, file_name: str, cfg: Config) -> dict:
    mode      = st.session_state.get("agent_mode_override", cfg.agent_mode)
    spark_ver = st.session_state.get("spark_version_override", cfg.spark_version)

    if mode == "multi":
        orchestrator = Orchestrator(cfg, file_name=file_name)
        result = orchestrator.review(code, language, spark_version=spark_ver)
        result["mode"] = "multi"
    else:
        raw_result = review_code(code, language)
        score      = extract_score(raw_result["review"])
        cert       = get_certification(score, cfg.pass_threshold)
        result = {
            "mode":            "single",
            "composite_score": score,
            "certified":       cert["certified"],
            "language":        language,
            "agents":          {},
            "key_findings":    [],
            "corrected_code":  "",
            "cost":            {"totals": {"cost_usd": 0, "total_tokens": 0}},
            "raw_review":      raw_result["review"],
            "chunks":          1,
            "cache_hit":       False,
            "cache_exact":     False,
        }

    result["language"]  = language
    result["file_name"] = file_name
    result["score"]     = result["composite_score"]

    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append(result)
    st.session_state["show_result"] = result
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Results tab
# ══════════════════════════════════════════════════════════════════════════════
def render_results_tab(result: dict, cfg: Config):
    if not result:
        st.info("Submit code using the **Input** tab to see results here.")
        return

    score     = result.get("composite_score", 0)
    certified = result.get("certified", False)
    language  = result.get("language", "")
    mode      = result.get("mode", "multi")

    # ── Score banner ────────────────────────────────────────────────────────
    css_class = "score-certified" if certified else "score-not-certified"
    badge     = "✅ CERTIFIED" if certified else "❌ NOT CERTIFIED"
    st.markdown(
        f'<div class="score-box {css_class}">'
        f'{badge}<br><span style="font-size:1.2rem">{score}/100</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Score",    f"{score}/100")
    col2.metric("Required", f"{cfg.pass_threshold}/100")
    col3.metric("Language", LANG_DISPLAY.get(language, language))
    col4.metric("Chunks",   result.get("chunks", 1))

    if result.get("cache_exact"):
        st.success("⚡ Exact cache hit — result from previous review of identical code.")
    elif result.get("cache_hit"):
        st.info("⚡ Cache hit — no LLM calls made for this review.")

    # ── Agent breakdown (multi mode) ────────────────────────────────────────
    agents  = result.get("agents", {})
    weights = result.get("weights", {})
    if mode == "multi" and agents:
        st.subheader("Agent Score Breakdown")
        cols = st.columns(len(agents))
        for col, (agent_key, agent_data) in zip(cols, agents.items()):
            label  = agent_key.replace("_agent", "").title()
            ascore = agent_data.get("score", 0)
            w      = weights.get(agent_key.replace("_agent", ""), 0)
            col.metric(f"{label} ({int(w*100)}%)", f"{ascore}/100")

        with st.expander("View Detailed Agent Reviews", expanded=False):
            for agent_key, agent_data in agents.items():
                label = agent_key.replace("_agent", "").title()
                st.markdown(f"### {label} Agent")
                st.markdown(agent_data.get("raw", "_No output_"))
                st.divider()

    # ── Single-agent review text ────────────────────────────────────────────
    if mode == "single" and "raw_review" in result:
        with st.expander("Full Review", expanded=True):
            st.markdown(result["raw_review"])

    # ── Key Findings ────────────────────────────────────────────────────────
    findings = result.get("key_findings", [])
    if findings:
        st.subheader("Key Findings")
        for finding in findings[:8]:
            st.markdown(f"- {finding}")

    # ── Corrected Code ──────────────────────────────────────────────────────
    corrected = result.get("corrected_code", "")
    if corrected:
        st.subheader("Corrected / Improved Code")
        st.code(corrected, language=_st_language(language))
        st.download_button(
            "⬇️ Download Corrected Code",
            corrected,
            file_name=f"corrected_{result.get('file_name', 'code')}",
            mime="text/plain",
        )

    # ── Download full report ────────────────────────────────────────────────
    report_json = json.dumps(result, indent=2, default=str)
    st.download_button(
        "⬇️ Download Full Report (JSON)",
        report_json,
        file_name="qa_spark_report.json",
        mime="application/json",
    )


def _st_language(lang: str) -> str:
    return {"pyspark": "python", "sparksql": "sql", "impala": "sql",
            "scala": "scala", "python": "python"}.get(lang, "text")


# ══════════════════════════════════════════════════════════════════════════════
# Cost & ROI tab
# ══════════════════════════════════════════════════════════════════════════════
def render_roi_tab(result: dict):
    st.header("Cost & ROI Dashboard")

    # ── Current review cost ─────────────────────────────────────────────────
    if result:
        st.subheader("Current Review Cost")
        cost        = result.get("cost", {})
        totals      = cost.get("totals", {})
        agents_cost = cost.get("agents", {})

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Cost (USD)",  f"${totals.get('cost_usd', 0):.4f}")
        col2.metric("Total Tokens",      f"{totals.get('total_tokens', 0):,}")
        col3.metric("Provider / Model",  f"{cost.get('provider','?')} / {cost.get('model','?')}")

        if agents_cost:
            st.markdown("**Per-Agent Breakdown**")
            rows = []
            for name, data in agents_cost.items():
                rows.append({
                    "Agent":             name.replace("_agent", "").title(),
                    "Prompt Tokens":     data.get("prompt_tokens", 0),
                    "Completion Tokens": data.get("completion_tokens", 0),
                    "Total Tokens":      data.get("total_tokens", 0),
                    "Cost (USD)":        f"${data.get('cost_usd', 0):.4f}",
                })
            st.table(rows)

        st.divider()

    # ── Lifetime ROI summary ────────────────────────────────────────────────
    st.subheader("Lifetime ROI Summary")
    summary = roi_summary()

    if summary.get("total_reviews", 0) == 0:
        st.info("No historical reviews found. Start reviewing code to build ROI data.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews",     summary["total_reviews"])
    col2.metric("Pass Rate",         f"{summary.get('pass_rate_pct', 0)}%")
    col3.metric("Avg Score",         f"{summary.get('avg_score', 0)}/100")
    col4.metric("Total Cost (USD)",  f"${summary.get('total_cost_usd', 0):.4f}")

    col5, col6, col7 = st.columns(3)
    col5.metric("Avg Cost / Review", f"${summary.get('avg_cost_usd', 0):.4f}")
    col6.metric("Certified Reviews", summary.get("certified", 0))
    col7.metric("Cache Hits",        summary.get("cache_hits", 0))

    by_lang = summary.get("by_language", {})
    if by_lang:
        st.subheader("Reviews by Language")
        lang_rows = []
        for lang, data in by_lang.items():
            lang_rows.append({
                "Language":   LANG_DISPLAY.get(lang, lang),
                "Reviews":    data["reviews"],
                "Avg Score":  f"{data['avg_score']}/100",
                "Cost (USD)": f"${data['cost_usd']:.4f}",
            })
        st.table(lang_rows)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    try:
        cfg = Config()
    except Exception as e:
        st.error(f"Configuration error: {e}")
        st.stop()

    render_sidebar(cfg)

    tab_input, tab_results, tab_roi = st.tabs([
        "📝 Input", "📊 Results", "💰 Cost & ROI"
    ])

    current_result = st.session_state.get("show_result")

    with tab_input:
        code, language, file_name = render_input_tab(cfg)
        if code:
            with st.spinner("Running review…"):
                try:
                    current_result = run_review(code, language, file_name, cfg)
                    st.success("Review complete! See the **Results** tab.")
                except Exception as e:
                    st.error(f"Review failed: {e}")
                    st.exception(e)

    with tab_results:
        render_results_tab(current_result, cfg)

    with tab_roi:
        render_roi_tab(current_result)


if __name__ == "__main__":
    main()
