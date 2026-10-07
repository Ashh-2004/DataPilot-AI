"""Analyzer agent: summarizes results, creates charts, and detects anomalies."""

from typing import Any
import logging
import json

import pandas as pd
from langchain_ollama import ChatOllama

from app.ml.alert_router import AlertRouter
from app.ml.anomaly_detector import AnomalyDetector
from app.services.domain_classifier import classify_domain


LOGGER = logging.getLogger(__name__)


class AnalyzerAgent:
    """Uses Ollama for insights and deterministic statistics for anomaly flags."""

    def __init__(self, model: str, base_url: str, webhook_url: str) -> None:
        self.model = model
        self.base_url = base_url
        self.llm = ChatOllama(model=model, base_url=base_url, temperature=0.2, num_ctx=2048, num_predict=500)
        self.anomaly_detector = AnomalyDetector()
        self.alert_router = AlertRouter()
        if webhook_url and not self.alert_router.webhook_url:
            self.alert_router.webhook_url = webhook_url

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return a conversational answer, insights, chart metadata, and anomaly status."""
        results = state.get("results", [])
        
        # If executor already generated a graceful fallback (all attempts failed / unanswerable), preserve it
        if not results and state.get("answer") and state.get("error"):
            return {
                "answer": state["answer"],
                "insight": state["answer"],
                "key_insights": state.get("key_insights", []),
                "limitations": state.get("limitations", []),
                "follow_up_questions": state.get("follow_up_questions", []),
                "chart": None,
                "anomaly": False,
                "anomaly_details": None,
                "anomaly_result": {},
            }
        results_df = pd.DataFrame(results)
        anomaly_result = self.anomaly_detector.detect(results_df)
        if anomaly_result["has_anomaly"]:
            self.alert_router.send(anomaly_result, state["question"])
        profile = self._profile(results, state.get("columns", []))
        analysis_rows = results[:30]
        overview = state.get("dataset_overview", {})
        summary = results_df.describe().to_string() if not results_df.empty else "No rows returned."
        
        is_overview = state.get("plan", {}).get("intent") == "dataset_overview"
        fallback = self._overview_fallback(overview)
        if is_overview:
            prompt_text = (
                "You are an expert AI Data Scientist like Grok or ChatGPT.\n"
                "The user is asking: 'What is this dataset about?'\n\n"
                "INSTRUCTIONS:\n"
                "Provide a comprehensive, beautifully structured dataset overview in Markdown format matching Grok's style.\n\n"
                "REQUIRED STRUCTURE:\n"
                "1. **Headline**: Bold title identifying the exact dataset domain (e.g., **This is a Sales & E-Commerce Transactions dataset**).\n"
                "2. **### Key Characteristics**: Bulleted list with Size (X rows × Y columns), Source/Domain style, and Main Purpose.\n"
                "3. **### Core Column Categories**: Group columns into logical categories.\n"
                "4. **### Concluding Summary**: A concise 2-sentence summary explaining what the dataset is used for.\n\n"
                "CRITICAL RULES:\n"
                "- Do NOT talk about SQL, database errors, or isolation forest anomaly scores.\n"
                "- Be helpful, clear, and professional like Grok.\n\n"
                f"Dataset Profile & Sample: {overview}\n"
            )
        else:
            prompt_text = (
                "You are an expert AI Data Scientist. Answer the user's question directly in conversational Markdown.\n"
                "Use the query results and dataset profile below to write a clear, natural language answer with key insights.\n"
                "Do NOT talk about SQL syntax or internal database execution.\n\n"
                f"User Question: {state['question']}\n"
                f"Rows Returned: {len(results)}\n"
                f"Query Results Summary:\n{summary}\n"
                f"Data Sample: {analysis_rows}\n"
            )

        try:
            response = self.llm.invoke(prompt_text)
            content = response.content if isinstance(response.content, str) else str(response.content)
            analysis = self._parse_analysis(content)
        except Exception as exc:
            LOGGER.warning("Analyzer LLM invocation failed or timed out: %s; using deterministic fallback", exc)
            analysis = fallback

        if is_overview:
            if not self._is_valid_overview_answer(analysis.get("answer", ""), overview):
                analysis["answer"] = fallback["answer"]
                analysis["key_insights"] = fallback["key_insights"]
                analysis["follow_up_questions"] = fallback["follow_up_questions"]
            elif not analysis.get("follow_up_questions"):
                analysis["follow_up_questions"] = fallback["follow_up_questions"]
        elif not analysis.get("answer") or self._is_clarification_response(analysis.get("answer", "")):
            analysis["answer"] = fallback["answer"]
            analysis["key_insights"] = fallback["key_insights"]
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
                    raw_answer = parsed.get("answer", content)
                    if isinstance(raw_answer, dict):
                        # Extract string if dict or dataframe-shaped object returned
                        answer_str = raw_answer.get("description") or raw_answer.get("text") or str(raw_answer)
                    else:
                        answer_str = str(raw_answer)

                    # Deduplicate key insights, limitations, follow-up questions
                    raw_insights = [str(item).strip() for item in parsed.get("key_insights", []) if item]
                    insights = list(dict.fromkeys([i for i in raw_insights if i and not i.startswith("{")]))

                    raw_limits = [str(item).strip() for item in parsed.get("limitations", []) if item]
                    limitations = list(dict.fromkeys([l for l in raw_limits if l]))

                    raw_questions = [str(item).strip() for item in parsed.get("follow_up_questions", []) if item]
                    follow_ups = list(dict.fromkeys([q for q in raw_questions if q]))

                    return {
                        "answer": answer_str,
                        "key_insights": insights[:5],
                        "limitations": limitations[:5],
                        "follow_up_questions": follow_ups[:3],
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
    def _generate_dynamic_follow_ups(overview: dict[str, Any]) -> list[str]:
        """Generate domain-agnostic follow-up questions from actual dataset columns."""
        tables = overview.get("tables", [])
        if not tables:
            return [
                "Can you provide summary statistics for this dataset?",
                "What are the main columns and data types?",
                "Are there any missing or duplicate values?",
            ]

        numeric_cols: list[str] = []
        categorical_cols: list[str] = []
        date_cols: list[str] = []

        for tbl in tables:
            for col in tbl.get("columns", []):
                cname = str(col.get("name", ""))
                ctype = str(col.get("type", "")).upper()
                if any(t in ctype for t in ("INT", "DOUBLE", "FLOAT", "DECIMAL", "NUMERIC", "BIGINT", "SMALLINT", "HUGEINT")):
                    if cname not in numeric_cols:
                        numeric_cols.append(cname)
                elif any(t in ctype for t in ("DATE", "TIMESTAMP", "TIME")):
                    if cname not in date_cols:
                        date_cols.append(cname)
                else:
                    if cname not in categorical_cols:
                        categorical_cols.append(cname)

        questions: list[str] = []
        if numeric_cols:
            questions.append(f"What is the distribution of {numeric_cols[0]}?")
        if categorical_cols and numeric_cols:
            questions.append(f"Which {categorical_cols[0]} has the highest {numeric_cols[0]}?")
        elif categorical_cols:
            questions.append(f"What are the top categories in {categorical_cols[0]}?")
        if date_cols and numeric_cols:
            questions.append(f"How has {numeric_cols[0]} changed over time by {date_cols[0]}?")
        elif date_cols:
            questions.append(f"What is the timeline range of {date_cols[0]}?")
        if len(numeric_cols) >= 2 and len(questions) < 3:
            questions.append(f"What is the relationship between {numeric_cols[0]} and {numeric_cols[1]}?")

        fallback_defaults = [
            "What are the top summary statistics across all columns?",
            "Are there any unusual outliers or anomalies in the data?",
            "Can you break down the dataset by key dimensions?",
        ]
        for default_q in fallback_defaults:
            if len(questions) < 3 and default_q not in questions:
                questions.append(default_q)

        return questions[:3]

    def _overview_fallback(self, overview: dict[str, Any]) -> dict[str, Any]:
        """Provide a rich Grok-style structured walkthrough of the dataset across all user tables using domain classifier."""
        tables = overview.get("tables", [])
        if not tables:
            return {
                "answer": "No dataset is currently loaded. Please upload a dataset in the sidebar to begin chatting with your data.",
                "key_insights": [],
                "follow_up_questions": AnalyzerAgent._generate_dynamic_follow_ups(overview),
            }

        sections: list[str] = []
        insights: list[str] = []

        for table in tables:
            columns = table.get("columns", [])
            column_names = [str(column.get("name")) for column in columns]
            table_name = str(table.get("name"))
            row_count = table.get("row_count", 0)
            sample_rows = table.get("sample", [])

            dtypes = {str(c.get("name")): str(c.get("type")) for c in columns if "name" in c}
            domain_info = classify_domain(
                column_names, dtypes, sample_rows, model=getattr(self, "model", "llama3.2"), base_url=getattr(self, "base_url", "http://localhost:11434")
            )
            domain = domain_info.get("domain", table_name.replace("_", " ").title())
            purpose = domain_info.get("reason", "General exploratory data analysis, statistical profiling, and pattern discovery.")

            demographics = [c for c in column_names if any(k in c.lower() for k in ("age", "sex", "gender", "unit", "location", "country", "region", "state", "city"))]
            metrics_cols = [c for c in column_names if any(k in c.lower() for k in ("price", "amount", "cost", "revenue", "salary", "total", "score", "count", "value", "quantity"))]
            dates_cols = [c for c in column_names if any(k in c.lower() for k in ("date", "time", "year", "month", "created", "updated", "joined"))]
            other_cols = [c for c in column_names if c not in demographics + metrics_cols + dates_cols]

            col_cats: list[str] = []
            if demographics:
                col_cats.append("- **Demographics & Identification**: " + ", ".join([f"`{c}`" for c in demographics]))
            if metrics_cols:
                col_cats.append("- **Numeric & KPI Metrics**: " + ", ".join([f"`{c}`" for c in metrics_cols]))
            if dates_cols:
                col_cats.append("- **Dates & Timestamps**: " + ", ".join([f"`{c}`" for c in dates_cols]))
            if other_cols:
                col_cats.append("- **Attributes & Descriptors**: " + ", ".join([f"`{c}`" for c in other_cols[:12]]))

            col_breakdown = "\n".join(col_cats) if col_cats else ", ".join([f"`{c}`" for c in column_names[:15]])

            answer = (
                f"**This is a {domain} dataset** containing **{row_count:,} records** across **{len(column_names)} columns**.\n\n"
                f"### Key Characteristics\n"
                f"- **Dataset Table**: `{table_name}`\n"
                f"- **Size**: {row_count:,} rows × {len(column_names)} columns\n"
                f"- **Main Purpose**: {purpose}\n\n"
                f"### Core Column Breakdown & Categories\n"
                f"{col_breakdown}\n\n"
                f"### Summary\n"
                f"In short, the table `{table_name}` is a structured collection of {domain.lower()} data designed for analyzing trends, evaluating key metrics, and performing conversational queries."
            )
            sections.append(answer)
            insights.append(f"{domain} table `{table_name}` loaded with {row_count:,} records across {len(column_names)} columns.")

        return {
            "answer": "\n\n".join(sections),
            "key_insights": list(dict.fromkeys(insights))[:5],
            "follow_up_questions": domain_info.get("suggested_questions")[:3] if domain_info.get("suggested_questions") else AnalyzerAgent._generate_dynamic_follow_ups(overview),
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
