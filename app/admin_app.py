"""
Data Platform Code Review QA Agent — Admin Panel
Run with: streamlit run app/admin_app.py --server.port 8502
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import streamlit as st

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config
from audit.audit_logger import audit_summary, load_audit_records, export_training_data
from cost.roi_logger import roi_summary
from mcp_server.cloudera_mcp import get_mcp_server

PROMPTS_DIR = ROOT / "agent" / "prompts"
ENV_FILE    = ROOT / ".env"
DATA_DIR    = ROOT / "data"

st.set_page_config(
    page_title="Admin — DP Code Review QA Agent",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.admin-header {
    background: linear-gradient(135deg,#1a0a2e,#2d1b4e);
    padding:18px 28px; border-radius:10px; margin-bottom:20px;
    border-bottom:3px solid #7c3aed;
}
.admin-header h1 { color:#e9d5ff; font-size:1.5rem; margin:0 0 4px; }
.admin-header p  { color:#a78bfa; font-size:.82rem; margin:0; }

.stat-card {
    background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
    padding:16px; text-align:center;
}
.stat-num   { font-size:2rem; font-weight:700; color:#2d3748; }
.stat-label { font-size:.78rem; color:#718096; margin-top:2px; }

.section-label {
    font-size:.72rem; font-weight:600; color:#4a5568;
    text-transform:uppercase; letter-spacing:.8px; margin-bottom:6px;
}
</style>
""", unsafe_allow_html=True)


def admin_header():
    st.markdown("""
    <div class="admin-header">
      <h1>⚙️ Admin Panel — Data Platform Code Review QA Agent</h1>
      <p>Manage models, prompts, scoring, auditing, and system configuration</p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar nav
# ══════════════════════════════════════════════════════════════════════════════
PAGES = [
    "📊 Dashboard",
    "🤖 Model Configuration",
    "✏️ Prompt Editor",
    "⚖️ Scoring & Weights",
    "🔍 Audit Log",
    "🧠 Training Data",
    "💾 Cache Manager",
    "🔌 Connections",
    "⚙️ System Settings",
]

def render_nav():
    with st.sidebar:
        st.markdown("### ⚙️ Admin Panel")
        st.caption("Data Platform Code Review QA Agent")
        st.divider()
        page = st.radio("Navigate", PAGES, label_visibility="collapsed")
        st.divider()
        st.caption("🔗 [Back to main app](http://localhost:8501)")
        return page


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard
# ══════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    st.header("System Dashboard")
    audit  = audit_summary()
    roi    = roi_summary()
    cfg    = _load_config()

    # System health row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_stat_card(audit.get("total_sessions",0), "Total Reviews"), unsafe_allow_html=True)
    with c2:
        st.markdown(_stat_card(f"{audit.get('pass_rate_pct',0)}%", "Pass Rate"), unsafe_allow_html=True)
    with c3:
        st.markdown(_stat_card(f"${roi.get('total_cost_usd',0):.4f}", "Total Spend"), unsafe_allow_html=True)
    with c4:
        st.markdown(_stat_card(audit.get("training_pairs_available",0), "Training Pairs"), unsafe_allow_html=True)

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Active Configuration")
        st.table([
            {"Setting": "Provider",        "Value": cfg.provider},
            {"Setting": "Model",           "Value": cfg.model_name},
            {"Setting": "Pass Threshold",  "Value": f"{cfg.pass_threshold}/100"},
            {"Setting": "Default Spark",   "Value": cfg.spark_version},
            {"Setting": "Agent Mode",      "Value": cfg.agent_mode},
            {"Setting": "Max Tokens/Chunk","Value": cfg.max_tokens_per_chunk},
        ])

    with c2:
        st.subheader("Review Quality Distribution")
        if audit.get("by_quality"):
            for label, count in audit["by_quality"].items():
                pct = round(count / max(audit["total_sessions"], 1) * 100)
                color = {"excellent":"green","good":"blue","needs_improvement":"orange",
                         "poor":"red","critical":"darkred"}.get(label, "grey")
                st.markdown(f"**{label.replace('_',' ').title()}** — {count} ({pct}%)")
                st.progress(pct / 100)
        else:
            st.caption("No data yet.")

    st.divider()
    st.subheader("Recent Activity")
    records = load_audit_records(last_n=10)
    if records:
        rows = []
        for r in reversed(records):
            rows.append({
                "Time":     r.get("timestamp","")[:16].replace("T"," "),
                "File":     r.get("file_name","?")[:30],
                "Language": r.get("language","?"),
                "Score":    r.get("scores",{}).get("composite","?"),
                "Certified":  "✅" if r.get("scores",{}).get("certified") else "❌",
                "Cost":     f"${r.get('cost',{}).get('totals',{}).get('cost_usd',0):.4f}",
            })
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No review sessions recorded yet.")


def _stat_card(value, label):
    return (f'<div class="stat-card">'
            f'<div class="stat-num">{value}</div>'
            f'<div class="stat-label">{label}</div></div>')


# ══════════════════════════════════════════════════════════════════════════════
# Model Configuration
# ══════════════════════════════════════════════════════════════════════════════
def page_model_config():
    st.header("🤖 Model Configuration")
    st.caption("Changes saved here update the `.env` file and take effect on next review.")

    env = _read_env()
    changed = {}

    st.subheader("Provider")
    provider = st.selectbox("Provider", ["azure", "openai", "bedrock"],
                            index=["azure","openai","bedrock"].index(env.get("PROVIDER","azure")))
    changed["PROVIDER"] = provider

    st.divider()

    if provider == "azure":
        st.subheader("Azure OpenAI Settings")
        c1, c2 = st.columns(2)
        with c1:
            changed["AZURE_DEPLOYED_MODEL"] = st.text_input(
                "Deployed Model Name", value=env.get("AZURE_DEPLOYED_MODEL","BDF-GLB-GPT-5"))
            changed["AZURE_API_VERSION"] = st.text_input(
                "API Version", value=env.get("AZURE_API_VERSION","2024-06-01"))
            changed["AZURE_ENDPOINT_BASE"] = st.text_input(
                "Endpoint Base URL", value=env.get("AZURE_ENDPOINT_BASE",""))
        with c2:
            changed["AZURE_ACCOUNT_NAME"] = st.text_input(
                "Account Name", value=env.get("AZURE_ACCOUNT_NAME",""))
            changed["AZURE_TENANT_ID"] = st.text_input(
                "Tenant ID", value=env.get("AZURE_TENANT_ID",""))
            st.caption("Service Principal credentials managed securely in .env")

    elif provider == "openai":
        st.subheader("OpenAI / Compatible Proxy Settings")
        c1, c2 = st.columns(2)
        with c1:
            changed["MODEL_NAME"] = st.text_input(
                "Model Name", value=env.get("MODEL_NAME","gpt-4o"))
        with c2:
            changed["OPENAI_API_BASE"] = st.text_input(
                "API Base URL", value=env.get("OPENAI_API_BASE","https://api.openai.com/v1"))

    st.divider()
    st.subheader("Cost Override (optional)")
    st.caption("Override the default price table for internal/proxy pricing.")
    c1, c2 = st.columns(2)
    with c1:
        changed["COST_INPUT_PER_1K"] = st.text_input(
            "Input price (USD per 1K tokens)", value=env.get("COST_INPUT_PER_1K",""),
            placeholder="e.g. 0.015")
    with c2:
        changed["COST_OUTPUT_PER_1K"] = st.text_input(
            "Output price (USD per 1K tokens)", value=env.get("COST_OUTPUT_PER_1K",""),
            placeholder="e.g. 0.060")

    st.divider()
    if st.button("💾 Save Model Configuration", type="primary"):
        _write_env_partial(changed)
        st.success("✅ Configuration saved. Restart the main app to apply changes.")


# ══════════════════════════════════════════════════════════════════════════════
# Prompt Editor
# ══════════════════════════════════════════════════════════════════════════════
def page_prompt_editor():
    st.header("✏️ Prompt Editor")
    st.caption("Edit agent system prompts and language-specific JSON prompt files.")

    # Agent system prompts (inline Python — read from source files)
    tab_json, tab_agents = st.tabs(["📄 Language Prompt Files (JSON)", "🤖 Agent System Prompts"])

    with tab_json:
        prompt_files = list(PROMPTS_DIR.glob("*.json")) if PROMPTS_DIR.exists() else []
        if not prompt_files:
            st.warning(f"No prompt files found in {PROMPTS_DIR}")
            return

        selected = st.selectbox("Select language prompt file",
                                [f.name for f in prompt_files])
        selected_path = PROMPTS_DIR / selected

        try:
            content = selected_path.read_text(encoding="utf-8")
            parsed  = json.loads(content)
        except Exception as e:
            st.error(f"Error reading {selected}: {e}")
            return

        st.caption(f"Editing: `{selected_path}`")
        edited = st.text_area("JSON Content", value=json.dumps(parsed, indent=2),
                              height=480, key=f"prompt_{selected}")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Save", type="primary", use_container_width=True):
                try:
                    json.loads(edited)   # validate
                    _backup_file(selected_path)
                    selected_path.write_text(edited, encoding="utf-8")
                    st.success("✅ Saved. Backup created.")
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")
        with c2:
            st.download_button("⬇️ Download", edited,
                               file_name=selected, mime="application/json",
                               use_container_width=True)

    with tab_agents:
        agent_files = {
            "Security Agent":     ROOT / "agent" / "security_agent.py",
            "Performance Agent":  ROOT / "agent" / "performance_agent.py",
            "Practices Agent":    ROOT / "agent" / "practices_agent.py",
            "Project Context":    ROOT / "agent" / "project_context_agent.py",
        }
        agent_name = st.selectbox("Select agent", list(agent_files.keys()))
        path = agent_files[agent_name]
        if path.exists():
            content = path.read_text(encoding="utf-8")
            # Extract just the system prompt string constants for display
            import re
            prompts = re.findall(
                r'(SYSTEM_PROMPT\w*)\s*=\s*"""(.*?)"""',
                content, re.DOTALL
            )
            if prompts:
                for name, prompt_text in prompts:
                    with st.expander(f"**{name}**", expanded=True):
                        edited_prompt = st.text_area(
                            "", value=prompt_text.strip(),
                            height=300, key=f"ap_{agent_name}_{name}",
                            label_visibility="collapsed",
                        )
                        if st.button(f"💾 Save {name}", key=f"save_{name}"):
                            new_content = content.replace(
                                f'"""{prompt_text}"""',
                                f'"""\n{edited_prompt}\n"""',
                            )
                            _backup_file(path)
                            path.write_text(new_content, encoding="utf-8")
                            st.success(f"✅ {name} saved.")
            else:
                st.info("No SYSTEM_PROMPT constants found in this file.")


# ══════════════════════════════════════════════════════════════════════════════
# Scoring & Weights
# ══════════════════════════════════════════════════════════════════════════════
def page_scoring():
    st.header("⚖️ Scoring & Weights")
    st.caption("Adjust how much each agent contributes to the composite score per language.")

    from agent.orchestrator import WEIGHTS, DEFAULT_WEIGHTS
    env = _read_env()

    st.subheader("Pass Threshold")
    threshold = st.slider(
        "Minimum score required for certification",
        min_value=50, max_value=100,
        value=int(env.get("PASS_THRESHOLD", "95")), step=1,
    )
    st.caption(f"Currently: **{threshold}/100**. Code below this score is blocked from execution.")

    st.divider()
    st.subheader("Agent Weight Distribution by Language")
    st.caption("Weights must sum to 1.0. Changes here update the orchestrator configuration.")

    updated_weights = {}
    for lang, w in WEIGHTS.items():
        st.markdown(f"**{lang.upper()}** — {_lang_icon(lang)}")
        c1, c2, c3 = st.columns(3)
        sec  = c1.number_input("Security",    0.0, 1.0, w["security"],    0.05, key=f"sec_{lang}")
        perf = c2.number_input("Performance", 0.0, 1.0, w["performance"], 0.05, key=f"prf_{lang}")
        prac = c3.number_input("Practices",   0.0, 1.0, w["practices"],   0.05, key=f"prc_{lang}")
        total = round(sec + perf + prac, 2)
        if abs(total - 1.0) > 0.01:
            st.error(f"⚠️ Weights sum to {total} — must equal 1.0")
        else:
            st.success(f"✓ Sum = {total}")
            updated_weights[lang] = {"security": sec, "performance": perf, "practices": prac}

    st.divider()
    if st.button("💾 Save Threshold", type="primary"):
        _write_env_partial({"PASS_THRESHOLD": str(threshold)})
        st.success("✅ Pass threshold saved.")

    st.caption("⚠️ Weight changes require editing `agent/orchestrator.py` WEIGHTS dict. "
               "Use the Prompt Editor tab for full file editing.")


# ══════════════════════════════════════════════════════════════════════════════
# Audit Log
# ══════════════════════════════════════════════════════════════════════════════
def page_audit_log():
    st.header("🔍 Audit Log")

    summary = audit_summary()
    if summary.get("total_sessions", 0) == 0:
        st.info("No audit records yet. Completed reviews are logged automatically.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sessions",   summary["total_sessions"])
    c2.metric("Secrets Detected", summary["secrets_detected"])
    c3.metric("With Corrections", summary["with_corrections"])
    c4.metric("With Context",     summary["with_user_context"])

    st.divider()

    # Filters
    with st.expander("🔽 Filter Records", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        lang_filter  = fc1.selectbox("Language", ["All"] + list(summary.get("by_language",{}).keys()))
        cert_filter  = fc2.selectbox("Certification", ["All", "Certified only", "Not certified only"])
        score_filter = fc3.slider("Minimum score", 0, 100, 0)

    records = load_audit_records(
        last_n=500,
        language=None if lang_filter=="All" else lang_filter,
        certified_only=(cert_filter == "Certified only"),
        min_score=score_filter,
    )

    st.subheader(f"Records ({len(records)} shown)")
    rows = []
    for r in reversed(records[-100:]):
        scores = r.get("scores", {})
        code   = r.get("code", {})
        rows.append({
            "Timestamp":   r.get("timestamp","")[:19].replace("T"," "),
            "File":        r.get("file_name","?")[:35],
            "Language":    r.get("language","?"),
            "Score":       scores.get("composite","?"),
            "Certified":   "✅" if scores.get("certified") else "❌",
            "Security":    scores.get("per_agent",{}).get("security_agent","?"),
            "Performance": scores.get("per_agent",{}).get("performance_agent","?"),
            "Practices":   scores.get("per_agent",{}).get("practices_agent","?"),
            "Secrets":     "⚠️ Yes" if code.get("secrets_redacted") else "No",
            "Modified":    "✏️ Yes" if code.get("code_was_modified") else "No",
            "Cost":        f"${r.get('cost',{}).get('totals',{}).get('cost_usd',0):.4f}",
        })
    st.dataframe(rows, use_container_width=True)

    st.divider()

    # Drill into a specific record
    st.subheader("Inspect a Review Session")
    if records:
        idx = st.number_input("Record index (0 = oldest)", 0, len(records)-1, len(records)-1)
        r = records[idx]
        with st.expander("📋 Full Record"):
            # Show key fields without raw code (too long)
            display = {k: v for k, v in r.items() if k not in ("agent_outputs",)}
            st.json(display)
        with st.expander("📝 Original Code"):
            lang = r.get("language","text")
            st.code(r.get("code",{}).get("original",""), language=_st_lang(lang))
        with st.expander("✏️ Corrected Code"):
            st.code(r.get("code",{}).get("corrected","(none)"), language=_st_lang(r.get("language","text")))
        with st.expander("🤖 Agent Outputs"):
            for agent, output in r.get("agent_outputs",{}).items():
                st.markdown(f"**{agent.replace('_agent','').title()}**")
                st.markdown(output)
                st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# Training Data
# ══════════════════════════════════════════════════════════════════════════════
def page_training_data():
    st.header("🧠 Training Data Export")
    st.caption(
        "Before/after code pairs annotated with scores and quality labels. "
        "Use for fine-tuning or evaluating reviewer models."
    )

    summary = audit_summary()
    c1, c2, c3 = st.columns(3)
    c1.metric("Usable Training Pairs", summary.get("training_pairs_available", 0))
    c2.metric("Total Sessions",        summary.get("total_sessions", 0))
    c3.metric("With Corrections",      summary.get("with_corrections", 0))

    st.divider()
    st.subheader("Quality Distribution")
    for label, count in summary.get("by_quality", {}).items():
        st.markdown(f"**{label.replace('_',' ').title()}:** {count}")

    st.divider()
    st.subheader("Export Options")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**JSONL — Full Training Pairs**")
        st.caption("Each line: `{agent, input_code, output, score, language, quality}`")
        if st.button("📦 Generate Export", type="primary", use_container_width=True):
            path = export_training_data()
            data = Path(path).read_text()
            st.download_button("⬇️ Download training_export.jsonl", data,
                               file_name="training_export.jsonl",
                               mime="application/jsonl", use_container_width=True)

    with c2:
        st.markdown("**JSON — Full Audit Records**")
        st.caption("Complete before/after with all context, for custom processing")
        if st.button("📦 Export All Audit Records", use_container_width=True):
            records = load_audit_records()
            data = json.dumps(records, indent=2, default=str)
            st.download_button("⬇️ Download audit_records.json", data,
                               file_name="audit_records.json",
                               mime="application/json", use_container_width=True)

    st.divider()
    st.subheader("Low-Score Reviews (for human review)")
    st.caption("These are candidates for manual correction and training signal.")
    low = load_audit_records(min_score=0)
    low_score = [r for r in low if r.get("scores",{}).get("composite",100) < 60]
    if low_score:
        for r in low_score[-10:]:
            score = r.get("scores",{}).get("composite","?")
            lang  = r.get("language","?")
            fname = r.get("file_name","?")
            ts    = r.get("timestamp","")[:10]
            with st.expander(f"Score {score}/100 · {lang} · {fname} · {ts}"):
                st.code(r.get("code",{}).get("original",""), language=_st_lang(lang))
                st.markdown("**Findings:**")
                for agent, output in r.get("agent_outputs",{}).items():
                    st.caption(f"**{agent}:** {output[:300]}…")
    else:
        st.success("No low-score reviews found.")


# ══════════════════════════════════════════════════════════════════════════════
# Cache Manager
# ══════════════════════════════════════════════════════════════════════════════
def page_cache_manager():
    st.header("💾 Cache Manager")
    cache_file = DATA_DIR / "embedding_cache.json"

    if not cache_file.exists():
        st.info("Cache is empty. Reviews are cached after first run.")
        return

    try:
        entries = json.loads(cache_file.read_text())
    except Exception as e:
        st.error(f"Could not read cache: {e}")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Cached Entries", len(entries))
    c2.metric("Cache File Size", f"{cache_file.stat().st_size / 1024:.1f} KB")
    c3.metric("Max Entries", 200)

    st.divider()
    st.subheader("Cache Entries")
    rows = []
    for e in entries[-50:]:
        rows.append({
            "Fingerprint": e.get("fingerprint","?"),
            "Language":    e.get("language","?"),
            "Score":       e.get("score","?"),
            "Findings":    "; ".join(e.get("key_findings",[])[:2])[:60],
        })
    st.dataframe(rows, use_container_width=True)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Export Cache",
            json.dumps(entries, indent=2),
            file_name="embedding_cache.json", mime="application/json",
            use_container_width=True)
    with c2:
        if st.button("🗑️ Clear Cache", type="secondary", use_container_width=True):
            if st.session_state.get("confirm_clear_cache"):
                cache_file.write_text("[]")
                st.success("Cache cleared.")
                st.session_state["confirm_clear_cache"] = False
            else:
                st.session_state["confirm_clear_cache"] = True
                st.warning("Click again to confirm clearing the cache.")


# ══════════════════════════════════════════════════════════════════════════════
# Connections
# ══════════════════════════════════════════════════════════════════════════════
def page_connections():
    st.header("🔌 Connections & Integrations")
    env = _read_env()

    with st.expander("Hive / Impala (Cloudera Metadata)", expanded=True):
        c1, c2 = st.columns(2)
        changed = {}
        with c1:
            changed["HIVE_HOST"]     = st.text_input("Host", value=env.get("HIVE_HOST",""))
            changed["HIVE_PORT"]     = st.text_input("Port", value=env.get("HIVE_PORT","10000"))
            changed["HIVE_DATABASE"] = st.text_input("Default Database", value=env.get("HIVE_DATABASE","default"))
        with c2:
            changed["HIVE_AUTH"]     = st.selectbox("Auth", ["none","ldap","kerberos"],
                index=["none","ldap","kerberos"].index(env.get("HIVE_AUTH","none")))
            changed["HIVE_USER"]     = st.text_input("Username", value=env.get("HIVE_USER",""))
            changed["HIVE_PASSWORD"] = st.text_input("Password", type="password", value="")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Save Hive Config", use_container_width=True):
                to_save = {k: v for k, v in changed.items() if v}
                _write_env_partial(to_save)
                st.success("✅ Saved.")
        with c2:
            if st.button("🔌 Test Connection", use_container_width=True):
                mcp = get_mcp_server()
                health = mcp.health_check()
                if health["connected"]:
                    st.success(f"✓ Connected to {health['host']}")
                else:
                    st.error(f"✗ {health.get('reason','failed')}")

    with st.expander("GitLab Integration"):
        st.caption("GitLab credentials are entered per-review in the main UI. "
                   "For CI/CD, set these in GitLab CI/CD Variables.")
        st.code("""# GitLab CI variables to configure:
PROVIDER, AZURE_TENANT_ID, AZURE_SERVICE_PRINCIPAL
AZURE_SERVICE_PRINCIPAL_SECRET, AZURE_ACCOUNT_NAME
AZURE_DEPLOYED_MODEL, PASS_THRESHOLD, SPARK_VERSION""", language="bash")


# ══════════════════════════════════════════════════════════════════════════════
# System Settings
# ══════════════════════════════════════════════════════════════════════════════
def page_system_settings():
    st.header("⚙️ System Settings")
    env = _read_env()
    changed = {}

    st.subheader("Review Behavior")
    c1, c2 = st.columns(2)
    with c1:
        changed["PASS_THRESHOLD"]       = str(st.number_input("Pass Threshold", 50, 100,
            int(env.get("PASS_THRESHOLD","95"))))
        changed["SPARK_VERSION"]        = st.text_input("Default Spark Version",
            value=env.get("SPARK_VERSION","3.4"))
        changed["AGENT_MODE"]           = st.selectbox("Agent Mode",["multi","single"],
            index=0 if env.get("AGENT_MODE","multi")=="multi" else 1)
    with c2:
        changed["MAX_TOKENS_PER_CHUNK"] = str(st.number_input("Max Tokens Per Chunk",
            500, 10000, int(env.get("MAX_TOKENS_PER_CHUNK","2500")), step=100))
        changed["MAX_TOKENS_PER_REVIEW"]= str(st.number_input("Max Tokens Per Review",
            5000, 100000, int(env.get("MAX_TOKENS_PER_REVIEW","20000")), step=1000))

    st.divider()
    st.subheader("Data & Storage")
    data_files = list(DATA_DIR.glob("*")) if DATA_DIR.exists() else []
    if data_files:
        st.table([{
            "File": f.name,
            "Size": f"{f.stat().st_size/1024:.1f} KB",
            "Modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        } for f in data_files])
    else:
        st.caption("No data files yet.")

    st.divider()
    if st.button("💾 Save All Settings", type="primary"):
        _write_env_partial(changed)
        st.success("✅ Settings saved. Restart the app to apply token limit changes.")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def _load_config():
    try:
        return Config()
    except Exception:
        return None


def _read_env() -> dict:
    if not ENV_FILE.exists():
        return {}
    result = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip()
    return result


def _write_env_partial(updates: dict):
    """Update specific keys in .env without touching others."""
    current = _read_env()
    current.update({k: v for k, v in updates.items() if v != ""})
    ENV_FILE.parent.mkdir(exist_ok=True)
    lines = ["# Data Platform Code Review QA Agent — Environment Configuration",
             "# Managed by Admin Panel — Do not commit this file\n"]
    for k, v in current.items():
        lines.append(f"{k}={v}")
    ENV_FILE.write_text("\n".join(lines) + "\n")


def _backup_file(path: Path):
    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text(path.read_text())


def _lang_icon(lang):
    return {"impala":"🗄️","pyspark":"⚡","sparksql":"⚡","scala":"⚡","python":"🐍"}.get(lang,"📄")


def _st_lang(lang):
    return {"pyspark":"python","sparksql":"sql","impala":"sql",
            "scala":"scala","python":"python"}.get(lang, "text")


# ══════════════════════════════════════════════════════════════════════════════
# Router
# ══════════════════════════════════════════════════════════════════════════════
def main():
    admin_header()
    page = render_nav()
    if   "Dashboard"     in page: page_dashboard()
    elif "Model Config"  in page: page_model_config()
    elif "Prompt Editor" in page: page_prompt_editor()
    elif "Scoring"       in page: page_scoring()
    elif "Audit Log"     in page: page_audit_log()
    elif "Training"      in page: page_training_data()
    elif "Cache"         in page: page_cache_manager()
    elif "Connections"   in page: page_connections()
    elif "System"        in page: page_system_settings()


if __name__ == "__main__":
    main()
