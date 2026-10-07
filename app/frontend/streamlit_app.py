"""ChatGPT / Gemini-Style AI Data Scientist Client for DataPilot AI."""

import os
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="DataPilot AI | Conversational Data Scientist",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for ChatGPT/Gemini style aesthetics
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
    }
    .chat-card {
        background-color: #1e222d;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid #2a2f3d;
    }
    .metric-badge {
        background-color: #262730;
        border-radius: 8px;
        padding: 8px 12px;
        display: inline-block;
        margin-right: 8px;
    }
    .issue-card-high {
        background-color: #2d1e1e;
        border-left: 4px solid #ef4444;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .issue-card-medium {
        background-color: #2d261e;
        border-left: 4px solid #f59e0b;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .issue-card-low {
        background-color: #1e262d;
        border-left: 4px solid #3b82f6;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_upload" not in st.session_state:
    st.session_state.last_upload = None


def render_safe_table(df: pd.DataFrame) -> None:
    """Render dataframe safely using Streamlit dataframe or HTML table fallback if PyArrow is restricted."""
    if df.empty:
        return
    try:
        st.dataframe(df, use_container_width=True)
    except Exception:
        st.table(df.head(25))


# --- SIDEBAR: Dataset Upload & Health Inspector ---
with st.sidebar:
    st.title("🤖 DataPilot AI")
    st.caption("Local Conversational Data Scientist & BI Assistant")
    st.markdown("---")

    st.subheader("📁 Upload Dataset")
    uploaded = st.file_uploader(
        "Upload CSV, JSON, Excel, or Parquet",
        type=["csv", "json", "xlsx", "xls", "parquet"],
    )
    if uploaded is not None and st.button("🚀 Load into Engine", use_container_width=True):
        with st.spinner("Cleaning and profiling dataset..."):
            try:
                response = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (uploaded.name, uploaded.getvalue())},
                    timeout=60,
                )
                response.raise_for_status()
                upload_data = response.json()
                st.session_state.last_upload = upload_data
                st.success(f"Successfully loaded {upload_data['rows']:,} rows into `{upload_data['table']}`.")
            except requests.RequestException as exc:
                st.error(f"Upload failed: {exc}")

    if st.session_state.last_upload and st.session_state.last_upload.get("cleaning"):
        cleaning = st.session_state.last_upload["cleaning"]
        st.markdown("---")
        st.subheader("📊 Dataset Health")
        q_score = cleaning.get("clean_score", cleaning.get("quality_score", 100))
        st.progress(q_score / 100, text=f"Clean Data Quality: {q_score}/100")

        c1, c2 = st.columns(2)
        c1.metric("Clean Rows", f"{cleaning.get('rows_after', 0):,}")
        c2.metric("Dupes Removed", f"{cleaning.get('duplicates_removed', 0):,}")

        if cleaning.get("columns_renamed"):
            with st.expander("Column Mapping"):
                st.json(cleaning["columns_renamed"])

    st.markdown("---")
    if st.button("🗑️ Reset Database & Clear Old Datasets", use_container_width=True):
        try:
            requests.post(f"{API_URL}/reset", timeout=10)
        except Exception:
            pass
        st.session_state.messages = []
        st.session_state.last_upload = None
        st.success("Database and chat history cleared!")
        st.rerun()


# --- TABS: Chat Interface & Persistent Dataset Report ---
tab_chat, tab_report = st.tabs(["💬 Chat Assistant", "📊 Dataset Report"])

# Function to process user question
def process_question(user_prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    history_payload = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-8:]
    ]

    with st.spinner("Analyzing data & generating insights..."):
        try:
            res = requests.post(
                f"{API_URL}/query",
                json={"question": user_prompt, "history": history_payload},
                timeout=180,
            )
            res.raise_for_status()
            result: dict[str, Any] = res.json()

            answer_text = result.get("answer") or result.get("insight") or "No output generated."
            key_insights = result.get("key_insights", [])
            rows = result.get("results", [])
            df = pd.DataFrame(rows) if rows else None
            chart = result.get("chart")

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": answer_text,
                "data_frame": df,
                "chart": chart,
                "key_insights": key_insights,
            }
            st.session_state.messages.append(assistant_msg)

        except requests.RequestException as exc:
            err_msg = f"Query failed: {exc}"
            if exc.response is not None:
                err_msg += f" ({exc.response.text})"
            st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {err_msg}"})

    st.rerun()


# =============================================================================
# TAB 1: Conversational Chat Assistant
# =============================================================================
with tab_chat:
    st.title("💬 Conversational Data Assistant")
    st.caption("Ask questions in natural language, request dataset overviews, or search for trends and anomalies.")

    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(
                "Hello! I am your **AI Data Scientist**.\n\n"
                "Upload your dataset in the sidebar (e.g. Sales, Housing, HR records) and ask me anything:\n"
                "- *'What is this dataset about?'*\n"
                "- *'Show top 5 records by revenue'* \n"
                "- *'Identify anomalies or unusual values'* \n"
                "- *'What are the key statistical trends?'*"
            )

    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if "data_frame" in msg and msg["data_frame"] is not None:
                render_safe_table(msg["data_frame"])
            if "chart" in msg and msg["chart"]:
                chart = msg["chart"]
                df = msg.get("data_frame")
                if df is not None and chart.get("x") in df and chart.get("y") in df:
                    st.plotly_chart(px.bar(df, x=chart["x"], y=chart["y"]), use_container_width=True)
            if "key_insights" in msg and msg["key_insights"]:
                st.markdown("**💡 Key Insights:**")
                for insight in msg["key_insights"]:
                    st.markdown(f"- {insight}")

    prompt = st.chat_input("Ask a question about your data...")
    if prompt and prompt.strip():
        process_question(prompt.strip())


# =============================================================================
# TAB 2: Persistent Dataset Report
# =============================================================================
with tab_report:
    st.header("📊 Persistent Dataset Report")
    st.caption("Automated schema profiling, domain identification, quality score comparison, and issue remediations.")

    # Fetch available datasets from API
    datasets_list = []
    try:
        d_res = requests.get(f"{API_URL}/datasets", timeout=10)
        if d_res.ok:
            datasets_list = d_res.json()
    except Exception:
        pass

    if not datasets_list and not st.session_state.last_upload:
        st.info("ℹ️ No dataset loaded yet. Please upload a dataset using the sidebar to view its comprehensive report.")
    else:
        table_names = [d["table_name"] for d in datasets_list] if datasets_list else []
        default_tbl = st.session_state.last_upload["table"] if st.session_state.last_upload else (table_names[0] if table_names else "")

        selected_table = st.selectbox("Select Dataset Table", options=table_names or [default_tbl], index=0 if default_tbl in table_names else 0)

        if selected_table:
            report_data = None
            try:
                r_res = requests.get(f"{API_URL}/datasets/{selected_table}/report", timeout=15)
                if r_res.ok:
                    report_data = r_res.json()
            except Exception as exc:
                st.error(f"Failed to fetch report from API: {exc}")

            if report_data:
                q_info = report_data.get("quality", {})
                d_info = report_data.get("domain", {})
                prof_info = report_data.get("profile", {})
                columns_list = prof_info.get("columns", [])
                issues_list = q_info.get("issues", [])

                # Header metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Raw Quality Score", f"{q_info.get('raw_score', 0)}/100")
                m2.metric("Clean Quality Score", f"{q_info.get('clean_score', 0)}/100")
                m3.metric("Clean Rows", f"{report_data.get('rows_clean', 0):,}")
                m4.metric("Domain", d_info.get("domain", "Unknown"))

                st.markdown("---")

                # Section 1: Domain Classification & Suggested Questions
                st.subheader("🏷️ Domain Classification & Suggested Questions")
                st.markdown(f"**Domain:** `{d_info.get('domain', 'Unknown')}` ({d_info.get('confidence', 1.0):.0%} confidence)")
                st.markdown(f"**Reasoning:** {d_info.get('reason', '')}")
                if d_info.get("likely_kpis"):
                    st.markdown(f"**Likely KPIs:** {', '.join(d_info['likely_kpis'])}")
                if d_info.get("likely_dimensions"):
                    st.markdown(f"**Likely Dimensions:** {', '.join(d_info['likely_dimensions'])}")

                st.markdown("**Click a suggested question to run it in the Chat Assistant:**")
                suggested_qs = d_info.get("suggested_questions", [])
                q_cols = st.columns(min(5, max(1, len(suggested_qs))))
                for idx, q_text in enumerate(suggested_qs[:5]):
                    with q_cols[idx % len(q_cols)]:
                        if st.button(f"❓ {q_text}", key=f"sq_btn_{idx}"):
                            process_question(q_text)

                st.markdown("---")

                # Section 2: Quality Issues & Suggestions
                st.subheader("⚠️ Quality Issues List & Recommendations")
                if issues_list:
                    for issue in issues_list:
                        sev = str(issue.get("severity", "medium")).lower()
                        card_class = f"issue-card-{sev}"
                        col_target = f"`{issue['column']}`" if issue.get("column") else "Table-level"
                        st.markdown(
                            f"""
                            <div class="{card_class}">
                                <strong>[{sev.upper()}] Target: {col_target}</strong><br>
                                {issue.get('description', '')}<br>
                                <em>💡 Suggestion: {issue.get('suggestion', '')}</em>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.success("✅ No quality issues detected!")

                st.markdown("---")

                # Section 3: Column Profiles Table
                st.subheader("📋 Column Profiles & Statistics")
                if columns_list:
                    table_rows = []
                    for col in columns_list:
                        stats_str = ""
                        if col.get("min") is not None and col.get("max") is not None:
                            stats_str += f"Range: [{col['min']}..{col['max']}] "
                        if col.get("mean") is not None:
                            stats_str += f"Mean: {col['mean']} "
                        if col.get("top_5"):
                            top_vals = ", ".join([f"{v.get('value')} ({v.get('count')})" for v in col["top_5"][:3]])
                            stats_str += f"Top: {top_vals}"

                        flags = []
                        if col.get("is_candidate_id"):
                            flags.append("ID")
                        if col.get("is_constant"):
                            flags.append("Constant")

                        table_rows.append({
                            "Column": col.get("name"),
                            "Type": col.get("type"),
                            "Null %": f"{col.get('null_pct', 0.0):.1f}%",
                            "Distinct": col.get("distinct_count", 0),
                            "Statistics / Samples": stats_str or "-",
                            "Flags": ", ".join(flags) if flags else "-",
                        })
                    col_df = pd.DataFrame(table_rows)
                    render_safe_table(col_df)

                st.markdown("---")

                # Section 4: Download Self-Contained HTML Report
                st.subheader("📥 Download Full HTML Dataset Report")
                try:
                    html_res = requests.get(f"{API_URL}/datasets/{selected_table}/report.html", timeout=15)
                    if html_res.ok:
                        st.download_button(
                            label="📥 Download Self-Contained HTML Report",
                            data=html_res.text,
                            file_name=f"{selected_table}_dataset_report.html",
                            mime="text/html",
                            use_container_width=True,
                        )
                except Exception as exc:
                    st.warning(f"Could not prepare HTML download: {exc}")
