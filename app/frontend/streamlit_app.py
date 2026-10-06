"""Minimal Streamlit client for uploading data and querying DataPilot AI."""

import os
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="DataPilot AI", layout="wide")
st.title("DataPilot AI")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

uploaded = st.file_uploader("Seed DuckDB with CSV or JSON", type=["csv", "json"])
if uploaded is not None and st.button("Upload data"):
    try:
        response = requests.post(
            f"{API_URL}/upload",
            files={"file": (uploaded.name, uploaded.getvalue())},
            timeout=30,
        )
        response.raise_for_status()
        st.success(f"Loaded {response.json()['rows']} rows into `{response.json()['table']}`.")
    except requests.RequestException as exc:
        st.error(f"Upload failed: {exc}")

question = st.chat_input("Ask anything about your uploaded dataset")
if question and question.strip():
    try:
        with st.spinner("Planning, querying, and analyzing..."):
            response = requests.post(
                f"{API_URL}/query",
                json={"question": question, "history": st.session_state.chat_history},
                timeout=120,
            )
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        st.session_state.chat_history.extend(
            [{"role": "user", "content": question}, {"role": "assistant", "content": result.get("answer", "")}]
        )
        st.subheader("Answer")
        st.write(result.get("answer", result.get("insight", "No answer returned.")))
        if result.get("key_insights"):
            st.subheader("Key insights")
            for insight in result["key_insights"]:
                st.markdown(f"- {insight}")
        if result.get("limitations"):
            st.caption("Limitations: " + " ".join(result["limitations"]))
        if result.get("follow_up_questions"):
            st.subheader("You could also ask")
            for follow_up in result["follow_up_questions"]:
                st.markdown(f"- {follow_up}")
        if result.get("anomaly"):
            st.warning(f"Anomaly detected: {result.get('anomaly_details')}")
        rows = result.get("results", [])
        if rows:
            frame = pd.DataFrame(rows)
            st.dataframe(frame, use_container_width=True)
            chart = result.get("chart")
            if chart and chart.get("x") in frame and chart.get("y") in frame:
                st.plotly_chart(px.bar(frame, x=chart["x"], y=chart["y"]), use_container_width=True)
        anomaly_result = result.get("anomaly_result", {})
        st.subheader("🚨 Anomalies Detected by Isolation Forest")
        if anomaly_result.get("has_anomaly"):
            flagged_rows = anomaly_result.get("anomaly_rows", [])
            st.warning(
                f"{anomaly_result.get('anomaly_count', len(flagged_rows))} anomalous rows flagged:\n\n"
                + pd.DataFrame(flagged_rows).to_string(index=False)
            )
            anomaly_scores = anomaly_result.get("anomaly_scores", [])
            if anomaly_scores:
                score_frame = pd.DataFrame(
                    {"row": list(range(len(anomaly_scores))), "anomaly_score": anomaly_scores}
                )
                st.bar_chart(score_frame, x="row", y="anomaly_score")
        else:
            st.success("No anomalies detected")
        with st.expander("Technical details"):
            st.json(result.get("plan", {}))
    except requests.RequestException as exc:
        detail = ""
        if exc.response is not None:
            detail = f" ({exc.response.text})"
        st.error(f"Query failed: {exc}{detail}")
