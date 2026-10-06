"""Analyzer agent: summarizes results, creates charts, and detects anomalies."""

from typing import Any
import logging
import json

import pandas as pd
from langchain_ollama import ChatOllama

from app.ml.alert_router import AlertRouter
from app.ml.anomaly_detector import AnomalyDetector


LOGGER = logging.getLogger(__name__)


class AnalyzerAgent:
    """Uses Ollama for insights and deterministic statistics for anomaly flags."""

    def __init__(self, model: str, base_url: str, webhook_url: str) -> None:
        self.llm = ChatOllama(model=model, base_url=base_url, temperature=0.2, num_ctx=2048, num_predict=256)
        self.anomaly_detector = AnomalyDetector()
        self.alert_router = AlertRouter()
        if webhook_url and not self.alert_router.webhook_url:
            self.alert_router.webhook_url = webhook_url

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return a conversational answer, insights, chart metadata, and anomaly status."""
        results = state.get("results", [])
        results_df = pd.DataFrame(results)
        anomaly_result = self.anomaly_detector.detect(results_df)
        if anomaly_result["has_anomaly"]:
            self.alert_router.send(anomaly_result, state["question"])
        profile = self._profile(results, state.get("columns", []))
        analysis_rows = results[:30]
        overview = state.get("dataset_overview", {})
        summary = results_df.describe().to_string() if not results_df.empty else "No rows returned."
        response = self.llm.invoke(
            "You are a helpful conversational data analyst. Answer the user's question directly. "
            "For a dataset walkthrough, explain what the dataset contains, its size, columns, "
            "important numeric ranges, notable patterns, and what users can ask next. "
            "Do not talk about SQL. Use the computed profile and sample below. Return JSON only with "
            "keys answer (string), key_insights (array of 2-5 concise strings), limitations (array of strings), "
            "and follow_up_questions (array of up to 3 useful questions). Do not invent facts. "
            f"User question: {state['question']}\n"
            f"Rows returned: {len(results)}\nProfile: {profile}\nSample: {analysis_rows}"
            f"\nDataset metadata and bounded samples: {overview}\n"
            f"You are a data analyst. The user asked: {state['question']}. "
            f"Query results summary: {summary}. "
            f"Anomalies detected: {anomaly_result['anomaly_count']} rows flagged. "
            f"Key columns involved: {anomaly_result['anomaly_columns']}. "
            "Write a 3-sentence business insight. Be specific about the anomaly if one exists, "
            "otherwise summarize the trend."
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        analysis = self._parse_analysis(content)
        if state.get("plan", {}).get("intent") == "dataset_overview":
            fallback = self._overview_fallback(overview)
            if self._is_clarification_response(analysis["answer"]):
                analysis["answer"] = fallback["answer"]
            if not analysis["key_insights"]:
                analysis["key_insights"] = fallback["key_insights"]
            if not analysis["follow_up_questions"]:
                analysis["follow_up_questions"] = fallback["follow_up_questions"]
        chart = self._chart_spec(results, state.get("columns", []))
        return {
            "answer": analysis["answer"],
            "insight": analysis["answer"],
            "key_insights": analysis["key_insights"],
            "limitations": analysis["limitations"],
            "follow_up_questions": analysis["follow_up_questions"],
            "chart": chart,
            "anomaly": anomaly_result["has_anomaly"],
            "anomaly_details": anomaly_result,
            "anomaly_result": anomaly_result,
        }

    @staticmethod
    def _profile(results: list[dict[str, Any]], columns: list[str]) -> dict[str, Any]:
        """Create compact deterministic facts so the model reasons over the full result shape."""
        profile: dict[str, Any] = {"row_count": len(results), "columns": columns}
        numeric: dict[str, dict[str, float]] = {}
        for column in columns:
            values = [
                float(row[column])
                for row in results
                if isinstance(row.get(column), (int, float)) and not isinstance(row.get(column), bool)
            ]
            if values:
                numeric[column] = {
                    "min": min(values),
                    "max": max(values),
                    "average": sum(values) / len(values),
                }
        profile["numeric_summary"] = numeric
        return profile

    @staticmethod
    def _parse_analysis(content: str) -> dict[str, Any]:
        """Parse structured model output while retaining a useful fallback for imperfect responses."""
        start, end = content.find("{"), content.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(content[start : end + 1])
                if isinstance(parsed, dict):
                    return {
                        "answer": str(parsed.get("answer", content)),
                        "key_insights": [str(item) for item in parsed.get("key_insights", [])][:5],
                        "limitations": [str(item) for item in parsed.get("limitations", [])][:5],
                        "follow_up_questions": [str(item) for item in parsed.get("follow_up_questions", [])][:3],
                    }
            except json.JSONDecodeError:
                pass
        return {
            "answer": content,
            "key_insights": [],
            "limitations": [],
            "follow_up_questions": [],
        }

    @staticmethod
    def _is_clarification_response(answer: str) -> bool:
        """Detect when the model ignored a broad overview request."""
        normalized = answer.casefold()
        return any(
            phrase in normalized
            for phrase in (
                "didn't specify",
                "did not specify",
                "please let me know which question",
                "which question you would like",
            )
        )

    @staticmethod
    def _overview_fallback(overview: dict[str, Any]) -> dict[str, Any]:
        """Provide a grounded walkthrough when the local model declines to summarize."""
        tables = overview.get("tables", [])
        if not tables:
            return {
                "answer": "The dataset is loaded, but no table metadata is available.",
                "key_insights": [],
                "follow_up_questions": [],
            }
        descriptions: list[str] = []
        insights: list[str] = []
        for table in tables:
            columns = table.get("columns", [])
            column_names = [str(column.get("name")) for column in columns]
            descriptions.append(
                f"`{table.get('name')}` contains {table.get('row_count')} rows and "
                f"{len(column_names)} columns: {', '.join(column_names)}."
            )
            numeric_columns = [
                str(column.get("name"))
                for column in columns
                if any(token in str(column.get("type", "")).upper() for token in ("INT", "DOUBLE", "DECIMAL", "FLOAT"))
            ]
            if numeric_columns:
                insights.append("Numeric analysis is available for: " + ", ".join(numeric_columns[:8]) + ".")
        return {
            "answer": "Here is a walkthrough of the uploaded dataset:\n\n" + "\n".join(descriptions),
            "key_insights": insights[:5],
            "follow_up_questions": [
                "What are the most important patterns in the numeric columns?",
                "Which groups or categories differ the most?",
                "Can you identify unusual values or anomalies?",
            ],
        }

    @staticmethod
    def _chart_spec(results: list[dict[str, Any]], columns: list[str]) -> dict[str, Any] | None:
        """Build a simple chart spec from the first categorical and numeric columns."""
        if not results or len(columns) < 2:
            return None
        numeric = next(
            (
                column
                for column in columns
                if any(isinstance(row.get(column), (int, float)) for row in results)
            ),
            None,
        )
        category = next((column for column in columns if column != numeric), None)
        return {"type": "bar", "x": category, "y": numeric} if numeric and category else None
