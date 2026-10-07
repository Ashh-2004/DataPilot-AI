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
        # Fallback for systems blocking PyArrow C-extension DLLs
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
        with st.spinner("Cleaning and indexing dataset..."):
            try:
                response = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (uploaded.name, uploaded.getvalue())},
                    timeout=45,
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
        q_score = cleaning.get("quality_score", 100)
        st.progress(q_score / 100, text=f"Data Quality Score: {q_score}/100")
        
        c1, c2 = st.columns(2)
        c1.metric("Clean Rows", f"{cleaning.get('rows_after', 0):,}")
        c2.metric("Duplicates Removed", f"{cleaning.get('duplicates_removed', 0):,}")

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


# --- MAIN CONTENT AREA: Conversational Chat Interface ---
st.title("💬 Conversational Data Assistant")
st.caption("Ask questions in natural language, request dataset overviews, or search for trends and anomalies.")

# Initial Welcome Message if Chat is Empty
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(
            "Hello! I am your **AI Data Scientist**.\n\n"
            "Upload your dataset in the sidebar (e.g. COVID-19 tracker, Housing prices, E-commerce sales) and ask me anything:\n"
            "- *'What is this dataset about?'*\n"
            "- *'Show top 5 records by revenue'* \n"
            "- *'Identify anomalies or unusual values'* \n"
            "- *'What are the key statistical trends?'*"
        )

# Render Chat History
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


# Function to process user question
def process_question(user_prompt: str) -> None:
    # Append user question
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Prepare chat history format for API backend
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
            anomaly_res = result.get("anomaly_result", {})

            # Format complete response content
            full_response_markdown = answer_text

            # Append assistant message
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": full_response_markdown,
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


# Chat Input Prompt
prompt = st.chat_input("Ask a question about your data...")
if prompt and prompt.strip():
    process_question(prompt.strip())
