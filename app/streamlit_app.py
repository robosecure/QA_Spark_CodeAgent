"""
Data Platform Code Review QA Agent
Streamlit UI — main application
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

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Platform Code Review QA Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

SPARK_LANGUAGES  = {"pyspark", "sparksql", "scala"}
LANG_DISPLAY = {
    "impala":   "Impala SQL",
    "pyspark":  "PySpark",
    "sparksql": "SparkSQL",
    "scala":    "Scala Spark",
    "python":   "Python",
}
LANG_ICONS = {
    "impala": "🗄️", "pyspark": "⚡", "sparksql": "⚡",
    "scala": "⚡", "python": "🐍",
}
CONFIDENCE_ICONS = {"high": "🟢", "medium": "🟡", "low": "🔴"}

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Brand header ── */
.dp-header {
    background: linear-gradient(135deg, #0a1628 0%, #1a3a5c 60%, #0d2137 100%);
    padding: 22px 32px 18px;
    border-radius: 12px;
    margin-bottom: 24px;
    border-bottom: 3px solid #1e90ff;
}
.dp-header h1 {
    color: #ffffff;
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0 0 4px 0;
    letter-spacing: -0.3px;
}
.dp-header .subtitle {
    color: #7fb3d3;
    font-size: 0.85rem;
    margin: 0;
}

/* ── Score boxes ── */
.score-wrap { border-radius: 14px; padding: 24px; text-align: center; margin-bottom: 20px; }
.score-certified   { background: linear-gradient(135deg,#1a472a,#2d6a4f);
                     border: 2px solid #40c074; color: #fff; }
.score-not-certified { background: linear-gradient(135deg,#5c1a1a,#8b2222);
                       border: 2px solid #e05252; color: #fff; }
.score-num  { font-size: 3.5rem; font-weight: 800; line-height: 1; }
.score-label { font-size: 1rem; opacity: 0.85; margin-top: 4px; }

/* ── Agent card ── */
.agent-card {
    background: #f8fafc; border-radius: 10px;
    padding: 16px; border-left: 5px solid #cbd5e0;
    margin-bottom: 12px;
}
.agent-card-high  { border-left-color: #38a169; }
.agent-card-mid   { border-left-color: #d69e2e; }
.agent-card-low   { border-left-color: #e53e3e; }

/* ── Badges ── */
.badge {
    display: inline-block;
    background: #ebf8ff; color: #2b6cb0;
    border: 1px solid #bee3f8; border-radius: 20px;
    padding: 3px 12px; font-size: 0.78rem;
    margin: 3px 4px 3px 0;
}
.badge-green { background:#f0fff4; color:#276749; border-color:#9ae6b4; }
.badge-orange{ background:#fffaf0; color:#7b341e; border-color:#fbd38d; }
.badge-red   { background:#fff5f5; color:#742a2a; border-color:#feb2b2; }

/* ── Context panel ── */
.context-panel {
    background: #f7fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 16px; margin-bottom: 16px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #0f1e2e; }
section[data-testid="stSidebar"] * { color: #c8d8e8 !important; }
section[data-testid="stSidebar"] .stButton > button {
    background: #1a3a5c !important; border: 1px solid #2a5a8c !important;
    color: #c8d8e8 !important; border-radius: 6px;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #2a5a8c !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════
def render_header():
    st.markdown("""
    <div class="dp-header">
      <h1>🛡️ Data Platform Code Review QA Agent</h1>
      <p class="subtitle">
        Automated multi-agent security, performance, and quality review
        for Cloudera CDP &nbsp;·&nbsp; Certification required before execution
      </p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar(cfg: Config):
    with st.sidebar:
        st.markdown("### 🛡️ DP Code Review")
        st.caption(f"v2.0 &nbsp;·&nbsp; {cfg.model_name}")
        st.divider()

        # ── History ───────────────────────────────────────────────────────
        st.markdown("**Recent Reviews**")
        history = st.session_state.get("history", [])
        if not history:
            st.caption("No reviews this session.")
        else:
            for i, entry in enumerate(reversed(history[-8:])):
                score = entry.get("score", "?")
                lang  = entry.get("language", "?")
                icon  = LANG_ICONS.get(lang, "📄")
                cert  = "✅" if entry.get("certified") else "❌"
                label = f"{cert} {icon} {LANG_DISPLAY.get(lang, lang.upper())} — {score}/100"
                if st.button(label, key=f"hist_{i}", use_container_width=True):
                    st.session_state["show_result"] = entry

        st.divider()

        # ── Spark Version (shown for all but hidden when irrelevant) ───────
        st.markdown("**Spark Version**")
        st.caption("Only applied when reviewing Spark code.")
        spark_ver = st.selectbox(
            "Spark Version", ["3.4", "3.5", "3.3", "3.2"], index=0,
            label_visibility="collapsed",
        )
        st.session_state["spark_version_override"] = spark_ver

        st.divider()

        # ── Cloudera Connection ────────────────────────────────────────────
        st.markdown("**🔌 Cloudera Metadata**")
        mcp = get_mcp_server()
        if mcp.is_available:
            health = mcp.health_check()
            if health["connected"]:
                st.success(f"✓ {health['host']}")
                with st.expander("Databases", expanded=False):
                    for db in health.get("databases", []):
                        st.caption(f"  {db}")
            else:
                st.error(health.get("reason", "Connection failed"))
        else:
            st.caption("Not connected")
            with st.expander("Connect to Cloudera"):
                st.code("HIVE_HOST=your-host\nHIVE_PORT=10000\n"
                        "HIVE_AUTH=ldap\nHIVE_USER=user\nHIVE_PASSWORD=pass",
                        language="bash")

        st.divider()
        st.caption("📊 [Admin Panel](http://localhost:8502)  ·  Pass threshold: "
                   f"**{cfg.pass_threshold}/100**")


# ══════════════════════════════════════════════════════════════════════════════
# Business Context Panel
# ══════════════════════════════════════════════════════════════════════════════
def render_context_panel() -> str:
    with st.expander("📋 Provide Business Context  *(optional — improves accuracy)*",
                     expanded=False):
        st.caption(
            "Tell the agents what this code is supposed to do. "
            "This is injected into every review prompt and prevents the AI from "
            "making incorrect assumptions about your use case."
        )
        c1, c2 = st.columns(2)
        with c1:
            purpose = st.text_area("What does this code do?",
                placeholder="e.g. Daily ETL that reads raw claims, applies HIPAA masking, "
                            "writes to gold layer...", height=90, key="ctx_purpose")
            volume = st.text_input("Expected data volume",
                placeholder="e.g. ~500M rows/day, ~200GB per run", key="ctx_volume")
            tables = st.text_area("Source / Target tables",
                placeholder="raw.claims (~2B rows, partitioned by claim_date)\n"
                            "gold.claims_masked — overwrite daily", height=90, key="ctx_tables")
        with c2:
            constraints = st.text_area("Constraints & SLAs",
                placeholder="e.g. Must complete in <2hr. 20-node cluster. "
                            "No broadcast joins > 50MB.", height=90, key="ctx_constraints")
            partitions = st.text_input("Current partitioning strategy",
                placeholder="e.g. year/month/day — ~3,000 partitions", key="ctx_partitions")
            notes = st.text_area("Additional notes for the reviewer",
                placeholder="e.g. Replacing legacy Impala query. Join on patient_id may be skewed.",
                height=90, key="ctx_notes")

    parts = []
    if purpose.strip():     parts.append(f"**Purpose:** {purpose.strip()}")
    if volume.strip():      parts.append(f"**Data Volume:** {volume.strip()}")
    if tables.strip():      parts.append(f"**Tables:**\n{tables.strip()}")
    if constraints.strip(): parts.append(f"**Constraints:** {constraints.strip()}")
    if partitions.strip():  parts.append(f"**Partitioning:** {partitions.strip()}")
    if notes.strip():       parts.append(f"**Notes:** {notes.strip()}")
    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Language selector — shows Spark version only when relevant
# ══════════════════════════════════════════════════════════════════════════════
def render_language_selector() -> str:
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        lang = st.selectbox(
            "Language / Dialect",
            SUPPORTED_LANGUAGES,
            format_func=lambda x: f"{LANG_ICONS.get(x,'')} {LANG_DISPLAY.get(x, x)}",
            key="language_select",
        )
    with c2:
        auto = st.checkbox("Auto-detect", value=True, key="auto_detect",
                           help="Detect language from file content and extension")
    with c3:
        if lang in SPARK_LANGUAGES:
            spark_ver = st.session_state.get("spark_version_override", "3.4")
            st.metric("Spark", spark_ver)

    return st.session_state.get("language_select", lang)


# ══════════════════════════════════════════════════════════════════════════════
# Input Tab
# ══════════════════════════════════════════════════════════════════════════════
def render_input_tab(cfg: Config):
    user_context    = render_context_panel()
    language        = render_language_selector()
    code, file_name, project_context = None, "<pasted>", ""

    tab_gl, tab_paste, tab_upload = st.tabs(
        ["🔗 GitLab / Repository", "📋 Paste Code", "📁 Upload File"]
    )

    # ── GitLab ─────────────────────────────────────────────────────────────
    with tab_gl:
        c1, c2 = st.columns(2)
        with c1:
            gl_url   = st.text_input("Project URL",
                placeholder="https://iqvia.gitlab-dedicated.com/rxcorp/bdf-admin/ecs-cml/my-project",
                key="gl_url")
            gl_token = st.text_input("Access Token", type="password", key="gl_token")
        with c2:
            gl_ref  = st.text_input("Branch / Tag / Commit", value="main", key="gl_ref")
            gl_path = st.text_input("File Path",
                placeholder="src/jobs/claims_etl.py", key="gl_path")

        with st.expander("📂 Scan entire repo for pipeline context", expanded=True):
            scan = st.checkbox("Scan repo before review (recommended)", value=True, key="scan_repo")
            scan_dir = st.text_input("Directory to scan (blank = root)",
                placeholder="src/jobs/", key="scan_dir") if scan else ""

        cb1, cb2 = st.columns(2)
        with cb1:
            fetch_btn = st.button("🔍 Fetch & Review", key="btn_gl", type="primary",
                                  use_container_width=True)
        with cb2:
            test_btn  = st.button("🔌 Test Connection", key="btn_gl_test",
                                  use_container_width=True)

        if test_btn:
            if gl_url and gl_token:
                with st.spinner("Testing connection…"):
                    ok, msg = _test_gitlab(gl_url, gl_token)
                st.success(f"✓ {msg}") if ok else st.error(f"✗ {msg}")

        if fetch_btn:
            if not (gl_url and gl_token and gl_path):
                st.error("Project URL, Access Token and File Path are required.")
            else:
                with st.spinner("Fetching file…"):
                    code, file_name = _fetch_gitlab_file(gl_url, gl_token, gl_ref, gl_path)
                if code:
                    if st.session_state.get("auto_detect"):
                        _apply_detection(detect_language(code, file_name))
                    if scan:
                        with st.spinner("Scanning repo for codebase context…"):
                            project_context = _build_project_context(
                                cfg, gl_url, gl_token, gl_ref, scan_dir, file_name)
                        if project_context:
                            st.success("✓ Codebase context built from repo scan.")

    # ── Paste ───────────────────────────────────────────────────────────────
    with tab_paste:
        pasted = st.text_area("", height=360,
            placeholder="Paste your SQL, Python, Scala, or PySpark code here…",
            key="paste_area", label_visibility="collapsed")
        if st.button("▶ Review Code", key="btn_paste", type="primary",
                     use_container_width=True):
            if pasted.strip():
                code = pasted
                if st.session_state.get("auto_detect"):
                    _apply_detection(detect_language(code))
            else:
                st.error("Please paste some code.")

    # ── Upload ──────────────────────────────────────────────────────────────
    with tab_upload:
        uploaded = st.file_uploader(
            "Drop a file or click to browse",
            type=["sql","py","scala","txt","hql"],
            label_visibility="visible",
        )
        if st.button("▶ Review File", key="btn_upload", type="primary",
                     use_container_width=True):
            if uploaded:
                code = uploaded.read().decode("utf-8", errors="replace")
                file_name = uploaded.name
                if st.session_state.get("auto_detect"):
                    _apply_detection(detect_language(code, file_name))
            else:
                st.error("Please upload a file.")

    # ── Detection badge ────────────────────────────────────────────────────
    det = st.session_state.get("last_detection")
    if det:
        icon = CONFIDENCE_ICONS.get(det["confidence"], "⚪")
        lang_label = LANG_DISPLAY.get(det["language"], det["language"])
        st.markdown(
            f'<span class="badge badge-green">'
            f'{icon} Auto-detected: <strong>{lang_label}</strong> '
            f'&nbsp;·&nbsp; {det["confidence"]} confidence via {det["method"]}'
            f'</span>',
            unsafe_allow_html=True,
        )

    language = st.session_state.get("language_select", language)
    return code, language, file_name, user_context, project_context


def _apply_detection(det: dict):
    lang = det.get("language")
    if lang and lang in SUPPORTED_LANGUAGES:
        st.session_state["language_select"] = lang
        st.session_state["last_detection"] = det


def _fetch_gitlab_file(project_url, token, ref, file_path):
    import requests, urllib.parse
    parts = project_url.rstrip("/").split("/")
    if len(parts) < 5:
        st.error("Invalid GitLab URL.")
        return None, "<unknown>"
    gitlab_base  = "/".join(parts[:3])
    namespace    = "/".join(parts[3:])
    enc_ns       = urllib.parse.quote_plus(namespace)
    enc_file     = urllib.parse.quote_plus(file_path)
    url  = f"{gitlab_base}/api/v4/projects/{enc_ns}/repository/files/{enc_file}/raw?ref={ref}"
    resp = requests.get(url, headers={"PRIVATE-TOKEN": token}, timeout=15)
    if resp.status_code == 200:
        return resp.text, file_path.split("/")[-1]
    st.error(f"GitLab {resp.status_code}: {resp.text[:200]}")
    return None, "<unknown>"


def _test_gitlab(project_url, token):
    import requests, urllib.parse
    parts = project_url.rstrip("/").split("/")
    if len(parts) < 5:
        return False, "Invalid URL"
    gitlab_base = "/".join(parts[:3])
    namespace   = "/".join(parts[3:])
    resp = requests.get(
        f"{gitlab_base}/api/v4/projects/{urllib.parse.quote_plus(namespace)}",
        headers={"PRIVATE-TOKEN": token}, timeout=10,
    )
    if resp.status_code == 200:
        d = resp.json()
        return True, f"{d.get('name_with_namespace','OK')} · {d.get('default_branch','main')}"
    return False, f"HTTP {resp.status_code}"


def _build_project_context(cfg, gl_url, gl_token, gl_ref, scan_path, primary_file):
    from agent.project_context_agent import ProjectContextAgent
    return ProjectContextAgent(cfg).build_from_gitlab(
        gl_url, gl_token, gl_ref, scan_path or "", primary_file)


# ══════════════════════════════════════════════════════════════════════════════
# Run Review
# ══════════════════════════════════════════════════════════════════════════════
def run_review(code, language, file_name, user_context, project_context, cfg) -> dict:
    spark_ver = st.session_state.get("spark_version_override", cfg.spark_version)
    # Auto mode: orchestrator always uses multi-agent; spark_version ignored for non-Spark
    effective_spark = spark_ver if language in SPARK_LANGUAGES else "n/a"

    orchestrator = Orchestrator(cfg, file_name=file_name)
    result = orchestrator.review(
        code, language,
        spark_version=effective_spark if effective_spark != "n/a" else cfg.spark_version,
        project_context=project_context,
        user_context=user_context,
    )
    result.update({
        "mode": "multi", "language": language,
        "file_name": file_name, "score": result["composite_score"],
    })
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
        st.markdown("""
        <div style="text-align:center; padding:60px; color:#718096;">
          <div style="font-size:3rem">🛡️</div>
          <h3>No review yet</h3>
          <p>Submit code from the <strong>Input</strong> tab to see results here.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    score     = result.get("composite_score", 0)
    certified = result.get("certified", False)
    language  = result.get("language", "")
    agents    = result.get("agents", {})
    weights   = result.get("weights", {})

    # ── Score hero ──────────────────────────────────────────────────────────
    css   = "score-certified" if certified else "score-not-certified"
    label = "✅ CERTIFIED FOR EXECUTION" if certified else "❌ NOT CERTIFIED — CHANGES REQUIRED"
    st.markdown(f"""
    <div class="score-wrap {css}">
      <div class="score-num">{score}<span style="font-size:1.5rem;opacity:.7">/100</span></div>
      <div class="score-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Summary metrics ────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Score",       f"{score}/100")
    c2.metric("Required",    f"{cfg.pass_threshold}/100")
    c3.metric("Gap",         f"{score - cfg.pass_threshold:+d}")
    c4.metric("Language",    f"{LANG_ICONS.get(language,'')} {LANG_DISPLAY.get(language, language)}")
    c5.metric("Chunks",      result.get("chunks", 1))

    # ── Context badges ─────────────────────────────────────────────────────
    badges = []
    if result.get("cache_exact"):
        badges.append(("badge-green", "⚡ Exact cache hit — instant result"))
    if result.get("project_context_used"):
        badges.append(("badge-green", "📂 Full codebase scanned"))
    if result.get("user_context"):
        badges.append(("badge-green", "📋 Business context included"))
    if result.get("mcp_used"):
        badges.append(("badge-green", "🔌 Live Cloudera metadata injected"))
    if badges:
        st.markdown(" ".join(
            f'<span class="badge {cls}">{txt}</span>' for cls, txt in badges
        ), unsafe_allow_html=True)

    st.markdown("")

    # ── Agent score breakdown ───────────────────────────────────────────────
    if agents:
        st.subheader("Agent Breakdown")
        cols = st.columns(len(agents))
        for col, (key, data) in zip(cols, agents.items()):
            agent_score = data.get("score", 0)
            label_name  = key.replace("_agent", "").title()
            w           = weights.get(key.replace("_agent", ""), 0)
            card_cls    = ("agent-card-high" if agent_score >= 80
                           else "agent-card-mid" if agent_score >= 60
                           else "agent-card-low")
            col.markdown(f"""
            <div class="agent-card {card_cls}">
              <div style="font-size:.8rem;opacity:.7;text-transform:uppercase;
                          letter-spacing:.5px">{label_name} &nbsp;·&nbsp; {int(w*100)}%</div>
              <div style="font-size:2rem;font-weight:700">{agent_score}<span
                style="font-size:1rem;opacity:.6">/100</span></div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("📄 Full Agent Review Details", expanded=False):
            for key, data in agents.items():
                st.markdown(f"### {key.replace('_agent','').title()} Agent Report")
                st.markdown(data.get("raw", "_No output._"))
                st.divider()

    # ── Key findings ────────────────────────────────────────────────────────
    findings = result.get("key_findings", [])
    if findings:
        st.subheader("Key Findings")
        f_cols = st.columns(min(2, len(findings)))
        for i, finding in enumerate(findings[:8]):
            f_cols[i % 2].markdown(f"▸ {finding}")

    st.markdown("")

    # ── Corrected code ──────────────────────────────────────────────────────
    corrected = result.get("corrected_code", "")
    if corrected:
        st.subheader("✏️ Corrected & Optimized Code")
        st.code(corrected, language=_st_lang(language))
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Download Corrected Code", corrected,
                file_name=f"corrected_{result.get('file_name','code')}",
                mime="text/plain", use_container_width=True)

    # ── Downloads ───────────────────────────────────────────────────────────
    st.subheader("Export")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Full Report (JSON)",
            json.dumps(result, indent=2, default=str),
            file_name="dp_review_report.json",
            mime="application/json", use_container_width=True)

    if result.get("user_context"):
        with st.expander("Business context that was provided"):
            st.markdown(result["user_context"])


def _st_lang(lang):
    return {"pyspark":"python","sparksql":"sql","impala":"sql",
            "scala":"scala","python":"python"}.get(lang, "text")


# ══════════════════════════════════════════════════════════════════════════════
# Cost & ROI Tab
# ══════════════════════════════════════════════════════════════════════════════
def render_roi_tab(result: dict):
    st.header("Cost & ROI")

    if result:
        st.subheader("This Review")
        totals = result.get("cost", {}).get("totals", {})
        agents_cost = result.get("cost", {}).get("agents", {})
        cost_meta   = result.get("cost", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Cost (USD)",   f"${totals.get('cost_usd',0):.4f}")
        c2.metric("Tokens Used",  f"{totals.get('total_tokens',0):,}")
        c3.metric("Model",        cost_meta.get("model", "—"))

        if agents_cost:
            st.table([{
                "Agent":    n.replace("_agent","").title(),
                "Input Tokens":  d.get("prompt_tokens",0),
                "Output Tokens": d.get("completion_tokens",0),
                "Total":    d.get("total_tokens",0),
                "USD":      f"${d.get('cost_usd',0):.4f}",
            } for n, d in agents_cost.items()])
        st.divider()

    st.subheader("Lifetime Summary")
    s = roi_summary()
    if s.get("total_reviews", 0) == 0:
        st.info("No completed reviews yet.")
        return

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Reviews",      s["total_reviews"])
    c2.metric("Pass Rate",    f"{s.get('pass_rate_pct',0)}%")
    c3.metric("Avg Score",    f"{s.get('avg_score',0)}/100")
    c4.metric("Total Spend",  f"${s.get('total_cost_usd',0):.4f}")

    c5,c6,c7 = st.columns(3)
    c5.metric("Avg / Review", f"${s.get('avg_cost_usd',0):.4f}")
    c6.metric("Certified",    s.get("certified",0))
    c7.metric("Cache Hits",   s.get("cache_hits",0))

    if s.get("by_language"):
        st.subheader("By Language")
        st.table([{
            "Language":  LANG_DISPLAY.get(l,l),
            "Reviews":   d["reviews"],
            "Avg Score": f"{d['avg_score']}/100",
            "Spend":     f"${d['cost_usd']:.4f}",
        } for l,d in s["by_language"].items()])


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    try:
        cfg = Config()
    except Exception as e:
        st.error(f"Configuration error: {e}")
        st.stop()

    render_header()
    render_sidebar(cfg)

    tab_input, tab_results, tab_roi = st.tabs(
        ["📝 Submit Code", "📊 Review Results", "💰 Cost & ROI"]
    )
    current = st.session_state.get("show_result")

    with tab_input:
        code, language, file_name, user_ctx, proj_ctx = render_input_tab(cfg)
        if code:
            with st.spinner("Running review — all three agents are analyzing your code…"):
                try:
                    current = run_review(code, language, file_name, user_ctx, proj_ctx, cfg)
                    st.success("✅ Review complete — see **Review Results** tab.")
                    st.session_state["active_tab"] = 1
                except Exception as e:
                    st.error(f"Review failed: {e}")
                    st.exception(e)

    with tab_results:
        render_results_tab(current, cfg)

    with tab_roi:
        render_roi_tab(current)


if __name__ == "__main__":
    main()
