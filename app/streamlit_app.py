"""
QA Spark CodeAgent — Streamlit UI

Tabs:  📝 Input | 📊 Results | 💰 Cost & ROI
Sidebar: Review history | Settings | Cloudera MCP connection
"""
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config
from agent.orchestrator import Orchestrator
from agent.reviewer import review_code, SUPPORTED_LANGUAGES
from agent.scorer import extract_score, get_certification
from agent.language_detector import detect_language
from mcp_server.cloudera_mcp import get_mcp_server
from cost.roi_logger import roi_summary

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

st.markdown("""
<style>
.score-box { padding:20px; border-radius:12px; text-align:center;
             font-size:2.6rem; font-weight:bold; margin-bottom:16px; }
.score-certified     { background:#d4edda; color:#155724; border:2px solid #28a745; }
.score-not-certified { background:#f8d7da; color:#721c24; border:2px solid #dc3545; }
.context-badge { background:#e8f4f8; border-left:4px solid #0099cc;
                 padding:8px 12px; border-radius:4px; margin:6px 0; font-size:0.85rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar(cfg: Config):
    with st.sidebar:
        st.title("⚡ QA Spark CodeAgent")
        st.caption(f"Model: **{cfg.model_name}** | Mode: **{cfg.agent_mode.upper()}**")
        st.divider()

        # ── Review History ─────────────────────────────────────────────────
        st.subheader("Review History")
        history = st.session_state.get("history", [])
        if not history:
            st.caption("No reviews yet this session.")
        for i, entry in enumerate(reversed(history[-10:])):
            score = entry.get("score", "?")
            lang  = entry.get("language", "?").upper()
            cert  = "✅" if entry.get("certified") else "❌"
            if st.button(f"{cert} {lang} — {score}/100", key=f"hist_{i}", use_container_width=True):
                st.session_state["show_result"] = entry

        st.divider()

        # ── Settings ───────────────────────────────────────────────────────
        st.subheader("Settings")
        mode = st.selectbox("Agent Mode", ["multi", "single"],
                            index=0 if cfg.agent_mode == "multi" else 1)
        st.session_state["agent_mode_override"] = mode

        spark_ver = st.selectbox("Spark Version", ["3.4", "3.5", "3.3"], index=0)
        st.session_state["spark_version_override"] = spark_ver

        st.divider()

        # ── Cloudera MCP Connection ────────────────────────────────────────
        st.subheader("🔌 Cloudera Connection")
        mcp = get_mcp_server()
        if mcp.is_available:
            health = mcp.health_check()
            if health["connected"]:
                st.success(f"Connected: `{health['host']}`")
                if health.get("databases"):
                    with st.expander("Databases"):
                        for db in health["databases"]:
                            st.caption(f"• {db}")
            else:
                st.error(f"Failed: {health.get('reason', 'unknown')}")
        else:
            st.caption("Not configured. Set `HIVE_HOST` to enable live metadata.")
            with st.expander("How to connect"):
                st.code("""# Add to .env:
HIVE_HOST=your-cloudera-host
HIVE_PORT=10000         # Hive
# or 21050 for Impala
HIVE_AUTH=ldap
HIVE_USER=your_user
HIVE_PASSWORD=your_pass
HIVE_DATABASE=default""", language="bash")


# ══════════════════════════════════════════════════════════════════════════════
# Business Context Panel (shown above code input)
# ══════════════════════════════════════════════════════════════════════════════
def render_context_panel() -> dict:
    """
    Interactive panel for developer to provide business context.
    Returns a dict of context values.
    """
    with st.expander("📋 Provide Business Context (optional but recommended)", expanded=False):
        st.caption("Help the agents understand what your code is supposed to do. "
                   "This prevents incorrect assumptions and improves accuracy.")

        col1, col2 = st.columns(2)
        with col1:
            purpose = st.text_area(
                "What does this code do?",
                placeholder="e.g. Daily ETL that reads raw claims data from S3, "
                            "applies HIPAA-compliant masking, and writes to the gold layer.",
                height=90, key="ctx_purpose",
            )
            data_volume = st.text_input(
                "Expected data volume",
                placeholder="e.g. ~500M rows/day, ~200GB per partition",
                key="ctx_volume",
            )
            tables_info = st.text_area(
                "Source / Target tables (and approximate sizes)",
                placeholder="e.g. raw.claims (~2B rows, partitioned by claim_date)\n"
                            "gold.claims_masked (target, overwrite daily)",
                height=90, key="ctx_tables",
            )
        with col2:
            constraints = st.text_area(
                "Known constraints or requirements",
                placeholder="e.g. Must complete in <2hr SLA. Cluster has 20 nodes. "
                            "Cannot use broadcast joins > 50MB due to memory limits.",
                height=90, key="ctx_constraints",
            )
            partition_info = st.text_input(
                "Current partitioning strategy",
                placeholder="e.g. Partitioned by year/month/day. ~3000 partitions.",
                key="ctx_partitions",
            )
            special_notes = st.text_area(
                "Anything else the reviewer should know",
                placeholder="e.g. This replaces an old Impala query. "
                            "We suspect the join on patient_id is skewed.",
                height=90, key="ctx_notes",
            )

    # Assemble into a single context string
    parts = []
    if purpose.strip():       parts.append(f"**Purpose:** {purpose.strip()}")
    if data_volume.strip():   parts.append(f"**Data Volume:** {data_volume.strip()}")
    if tables_info.strip():   parts.append(f"**Tables:**\n{tables_info.strip()}")
    if constraints.strip():   parts.append(f"**Constraints:** {constraints.strip()}")
    if partition_info.strip():parts.append(f"**Partitioning:** {partition_info.strip()}")
    if special_notes.strip(): parts.append(f"**Notes:** {special_notes.strip()}")

    return {
        "user_context": "\n\n".join(parts),
        "has_context": bool(parts),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Input Tab
# ══════════════════════════════════════════════════════════════════════════════
def render_input_tab(cfg: Config):
    st.header("Submit Code for Review")

    # Business context panel (always visible above input)
    ctx = render_context_panel()

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        language = st.selectbox(
            "Language / Dialect",
            SUPPORTED_LANGUAGES,
            format_func=lambda x: LANG_DISPLAY.get(x, x),
            key="language_select",
        )
    with col2:
        st.metric("Pass Threshold", f"{cfg.pass_threshold}/100")
    with col3:
        auto_detect = st.checkbox("Auto-detect language", value=True, key="auto_detect")

    sub_gitlab, sub_paste, sub_upload = st.tabs([
        "🔗 GitLab Repo", "📋 Paste Code", "📁 Upload File"
    ])

    code = None
    file_name = "<pasted>"
    project_context = ""
    gitlab_creds = {}

    # ── GitLab tab ─────────────────────────────────────────────────────────
    with sub_gitlab:
        col_a, col_b = st.columns(2)
        with col_a:
            gl_url   = st.text_input("GitLab Project URL",
                placeholder="https://iqvia.gitlab-dedicated.com/rxcorp/bdf-admin/ecs-cml/my-project")
            gl_token = st.text_input("GitLab Access Token", type="password")
        with col_b:
            gl_ref   = st.text_input("Branch / Tag", value="main")
            gl_path  = st.text_input("File Path in Repo",
                placeholder="src/jobs/claims_etl.py")

        scan_repo = st.checkbox(
            "📂 Scan entire repo for codebase context (recommended)",
            value=True, key="scan_repo",
            help="Fetches all related files so agents understand the full pipeline, "
                 "not just the single file being reviewed."
        )
        repo_path = st.text_input(
            "Repo subfolder to scan (leave blank for root)",
            placeholder="src/jobs/", key="repo_scan_path",
        ) if scan_repo else ""

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Fetch & Review", key="btn_gitlab", type="primary"):
                if not (gl_url and gl_token and gl_path):
                    st.error("Please fill in Project URL, Token, and File Path.")
                else:
                    with st.spinner("Fetching file…"):
                        code, file_name = _fetch_gitlab_file(gl_url, gl_token, gl_ref, gl_path)
                    if code and auto_detect:
                        detected = detect_language(code, file_name)
                        _apply_detected_language(detected)
                    if code and scan_repo:
                        gitlab_creds = {"url": gl_url, "token": gl_token, "ref": gl_ref}
                        with st.spinner("Scanning codebase for context…"):
                            project_context = _build_project_context(
                                cfg, gl_url, gl_token, gl_ref,
                                repo_path or "", file_name
                            )
                        if project_context:
                            st.success("✅ Codebase context built from repo scan.")
        with col_btn2:
            if st.button("Test Connection", key="btn_gitlab_test"):
                if gl_url and gl_token:
                    with st.spinner("Testing…"):
                        ok, msg = _test_gitlab_connection(gl_url, gl_token)
                    if ok:
                        st.success(f"Connected: {msg}")
                    else:
                        st.error(f"Failed: {msg}")

    # ── Paste tab ───────────────────────────────────────────────────────────
    with sub_paste:
        pasted = st.text_area(
            "Paste your code here",
            height=340, placeholder="SELECT * FROM ...", key="paste_area",
        )
        if st.button("Review Pasted Code", key="btn_paste", type="primary"):
            if pasted.strip():
                code = pasted
                if auto_detect:
                    detected = detect_language(code)
                    _apply_detected_language(detected)
            else:
                st.error("Please paste some code first.")

    # ── Upload tab ──────────────────────────────────────────────────────────
    with sub_upload:
        uploaded = st.file_uploader(
            "Upload a code file",
            type=["sql", "py", "scala", "txt", "hql"],
            key="file_uploader",
        )
        if st.button("Review Uploaded File", key="btn_upload", type="primary"):
            if uploaded:
                code = uploaded.read().decode("utf-8", errors="replace")
                file_name = uploaded.name
                if auto_detect:
                    detected = detect_language(code, file_name)
                    _apply_detected_language(detected)
            else:
                st.error("Please upload a file first.")

    # Show detected language badge if auto-detect ran
    if st.session_state.get("last_detection"):
        d = st.session_state["last_detection"]
        conf_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(d["confidence"], "⚪")
        st.markdown(
            f'<div class="context-badge">'
            f'{conf_color} Auto-detected: <strong>{LANG_DISPLAY.get(d["language"], d["language"])}</strong> '
            f'({d["confidence"]} confidence, via {d["method"]})'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Read current language from selectbox (may have been updated by auto-detect)
    language = st.session_state.get("language_select", language)
    return code, language, file_name, ctx["user_context"], project_context


def _apply_detected_language(detected: dict):
    lang = detected.get("language")
    if lang and lang in SUPPORTED_LANGUAGES:
        st.session_state["language_select"] = lang
        st.session_state["last_detection"] = detected


def _fetch_gitlab_file(project_url: str, token: str, ref: str, file_path: str):
    import requests, urllib.parse
    project_url = project_url.rstrip("/")
    parts = project_url.split("/")
    if len(parts) < 5:
        st.error("Invalid GitLab URL.")
        return None, "<unknown>"
    gitlab_base  = "/".join(parts[:3])
    namespace    = "/".join(parts[3:])
    encoded_ns   = urllib.parse.quote_plus(namespace)
    encoded_file = urllib.parse.quote_plus(file_path)
    url = f"{gitlab_base}/api/v4/projects/{encoded_ns}/repository/files/{encoded_file}/raw?ref={ref}"
    resp = requests.get(url, headers={"PRIVATE-TOKEN": token}, timeout=15)
    if resp.status_code == 200:
        return resp.text, file_path.split("/")[-1]
    st.error(f"GitLab error {resp.status_code}: {resp.text[:200]}")
    return None, "<unknown>"


def _test_gitlab_connection(project_url: str, token: str) -> tuple:
    import requests, urllib.parse
    project_url = project_url.rstrip("/")
    parts = project_url.split("/")
    if len(parts) < 5:
        return False, "Invalid URL format"
    gitlab_base = "/".join(parts[:3])
    namespace   = "/".join(parts[3:])
    encoded_ns  = urllib.parse.quote_plus(namespace)
    url  = f"{gitlab_base}/api/v4/projects/{encoded_ns}"
    resp = requests.get(url, headers={"PRIVATE-TOKEN": token}, timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        return True, f"{data.get('name_with_namespace', namespace)} — {data.get('default_branch', 'main')}"
    return False, f"HTTP {resp.status_code}"


def _build_project_context(cfg, gl_url, gl_token, gl_ref, scan_path, primary_file) -> str:
    from agent.project_context_agent import ProjectContextAgent
    pca = ProjectContextAgent(cfg)
    return pca.build_from_gitlab(gl_url, gl_token, gl_ref, scan_path, primary_file)


# ══════════════════════════════════════════════════════════════════════════════
# Run Review
# ══════════════════════════════════════════════════════════════════════════════
def run_review(code, language, file_name, user_context, project_context, cfg) -> dict:
    mode      = st.session_state.get("agent_mode_override", cfg.agent_mode)
    spark_ver = st.session_state.get("spark_version_override", cfg.spark_version)

    if mode == "multi":
        orchestrator = Orchestrator(cfg, file_name=file_name)
        result = orchestrator.review(
            code, language,
            spark_version=spark_ver,
            project_context=project_context,
            user_context=user_context,
        )
        result["mode"] = "multi"
    else:
        raw = review_code(code, language)
        score = extract_score(raw["review"])
        cert  = get_certification(score, cfg.pass_threshold)
        result = {
            "mode": "single", "composite_score": score,
            "certified": cert["certified"], "language": language,
            "agents": {}, "key_findings": [], "corrected_code": "",
            "cost": {"totals": {"cost_usd": 0, "total_tokens": 0}},
            "raw_review": raw["review"], "chunks": 1,
            "cache_hit": False, "cache_exact": False,
            "mcp_used": False, "project_context_used": bool(project_context),
            "user_context": user_context,
        }

    result.update({"language": language, "file_name": file_name,
                   "score": result["composite_score"]})

    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append(result)
    st.session_state["show_result"] = result
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Results Tab
# ══════════════════════════════════════════════════════════════════════════════
def render_results_tab(result: dict, cfg: Config):
    if not result:
        st.info("Submit code using the **Input** tab to see results here.")
        return

    score     = result.get("composite_score", 0)
    certified = result.get("certified", False)
    language  = result.get("language", "")
    mode      = result.get("mode", "multi")

    # Score banner
    css = "score-certified" if certified else "score-not-certified"
    badge = "✅ CERTIFIED" if certified else "❌ NOT CERTIFIED"
    st.markdown(
        f'<div class="score-box {css}">{badge}<br>'
        f'<span style="font-size:1.2rem">{score}/100</span></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score",    f"{score}/100")
    c2.metric("Required", f"{cfg.pass_threshold}/100")
    c3.metric("Language", LANG_DISPLAY.get(language, language))
    c4.metric("Chunks",   result.get("chunks", 1))

    # Context badges
    badges = []
    if result.get("cache_exact"):
        badges.append("⚡ Exact cache hit — no LLM calls needed")
    if result.get("project_context_used"):
        badges.append("📂 Codebase context: full repo scanned")
    if result.get("user_context"):
        badges.append("📋 Business context: developer input included")
    if result.get("mcp_used"):
        badges.append("🔌 Live Cloudera metadata: table stats injected")
    for b in badges:
        st.markdown(f'<div class="context-badge">{b}</div>', unsafe_allow_html=True)

    # Agent breakdown
    agents  = result.get("agents", {})
    weights = result.get("weights", {})
    if mode == "multi" and agents:
        st.subheader("Agent Score Breakdown")
        cols = st.columns(len(agents))
        for col, (key, data) in zip(cols, agents.items()):
            label = key.replace("_agent", "").title()
            w = weights.get(key.replace("_agent", ""), 0)
            col.metric(f"{label} ({int(w*100)}%)", f"{data.get('score',0)}/100")

        with st.expander("View Detailed Agent Reviews", expanded=False):
            for key, data in agents.items():
                st.markdown(f"### {key.replace('_agent','').title()} Agent")
                st.markdown(data.get("raw", "_No output_"))
                st.divider()

    if mode == "single" and "raw_review" in result:
        with st.expander("Full Review", expanded=True):
            st.markdown(result["raw_review"])

    # Key findings
    findings = result.get("key_findings", [])
    if findings:
        st.subheader("Key Findings")
        for f in findings[:8]:
            st.markdown(f"- {f}")

    # Corrected code
    corrected = result.get("corrected_code", "")
    if corrected:
        st.subheader("Corrected / Improved Code")
        st.code(corrected, language=_st_lang(language))
        st.download_button("⬇️ Download Corrected Code", corrected,
                           file_name=f"corrected_{result.get('file_name','code')}",
                           mime="text/plain")

    # Developer context used (expandable for transparency)
    if result.get("user_context"):
        with st.expander("Business context that was provided", expanded=False):
            st.markdown(result["user_context"])

    st.download_button("⬇️ Download Full Report (JSON)",
                       json.dumps(result, indent=2, default=str),
                       file_name="qa_spark_report.json", mime="application/json")


def _st_lang(lang):
    return {"pyspark":"python","sparksql":"sql","impala":"sql",
            "scala":"scala","python":"python"}.get(lang, "text")


# ══════════════════════════════════════════════════════════════════════════════
# Cost & ROI Tab
# ══════════════════════════════════════════════════════════════════════════════
def render_roi_tab(result: dict):
    st.header("Cost & ROI Dashboard")

    if result:
        st.subheader("Current Review Cost")
        totals = result.get("cost", {}).get("totals", {})
        agents_cost = result.get("cost", {}).get("agents", {})
        cost_data = result.get("cost", {})

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Cost (USD)",  f"${totals.get('cost_usd',0):.4f}")
        c2.metric("Total Tokens",      f"{totals.get('total_tokens',0):,}")
        c3.metric("Provider / Model",  f"{cost_data.get('provider','?')} / {cost_data.get('model','?')}")

        if agents_cost:
            st.table([{
                "Agent":             n.replace("_agent","").title(),
                "Prompt Tokens":     d.get("prompt_tokens",0),
                "Completion Tokens": d.get("completion_tokens",0),
                "Total Tokens":      d.get("total_tokens",0),
                "Cost (USD)":        f"${d.get('cost_usd',0):.4f}",
            } for n, d in agents_cost.items()])
        st.divider()

    st.subheader("Lifetime ROI Summary")
    summary = roi_summary()
    if summary.get("total_reviews", 0) == 0:
        st.info("No historical reviews yet.")
        return

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Reviews",    summary["total_reviews"])
    c2.metric("Pass Rate",        f"{summary.get('pass_rate_pct',0)}%")
    c3.metric("Avg Score",        f"{summary.get('avg_score',0)}/100")
    c4.metric("Total Cost (USD)", f"${summary.get('total_cost_usd',0):.4f}")

    c5,c6,c7 = st.columns(3)
    c5.metric("Avg Cost / Review", f"${summary.get('avg_cost_usd',0):.4f}")
    c6.metric("Certified Reviews", summary.get("certified",0))
    c7.metric("Cache Hits",        summary.get("cache_hits",0))

    by_lang = summary.get("by_language", {})
    if by_lang:
        st.subheader("By Language")
        st.table([{
            "Language":   LANG_DISPLAY.get(l, l),
            "Reviews":    d["reviews"],
            "Avg Score":  f"{d['avg_score']}/100",
            "Cost (USD)": f"${d['cost_usd']:.4f}",
        } for l, d in by_lang.items()])


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

    tab_input, tab_results, tab_roi = st.tabs(["📝 Input", "📊 Results", "💰 Cost & ROI"])
    current_result = st.session_state.get("show_result")

    with tab_input:
        code, language, file_name, user_context, project_context = render_input_tab(cfg)
        if code:
            with st.spinner("Running multi-agent review…"):
                try:
                    current_result = run_review(
                        code, language, file_name,
                        user_context, project_context, cfg
                    )
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
