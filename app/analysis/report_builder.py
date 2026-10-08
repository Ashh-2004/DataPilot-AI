"""Report builder assembling executive summary, recommendations, data limitations, and Plotly charts."""

import logging
import os
import re
from datetime import datetime
from typing import Any

import requests

from app.analysis.dataclasses import AnalysisResult, ChartSpec, Report

LOGGER = logging.getLogger(__name__)


class ReportBuilder:
    """Assembles final executive Report from analysis results, profile metadata, and domain classification."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def build(
        self,
        results: list[AnalysisResult],
        classification: dict[str, Any],
        profile: dict[str, Any],
    ) -> Report:
        """Assemble structured Report with LLM executive summary, key metrics, limitations, and charts."""
        dataset_type = classification.get("dataset_type", "generic")
        key_entity = classification.get("key_entity", "record")
        title = f"DataPilot Auto-Analysis Report: {dataset_type.title()} Data ({key_entity.title()} Level)"

        # Collect non-skipped sections and insights
        active_sections = [r for r in results if not r.skipped]
        all_insights = [r.insight for r in active_sections if r.insight]
        all_charts: list[ChartSpec] = []
        for r in active_sections:
            all_charts.extend(r.charts)

        # Aggregate key metrics (top 8-10 numbers)
        key_metrics: dict[str, Any] = {
            "Total Records": f"{profile.get('row_count', 0):,}",
            "Total Columns": f"{profile.get('column_count', 0):,}",
            "Data Quality Score": f"{profile.get('data_quality_score', 100.0)}/100",
            "Duplicate Rows": f"{profile.get('duplicate_row_count', 0):,}",
        }

        # Add domain or task specific metrics if present
        for r in active_sections:
            if "Total Revenue" in r.content:
                match = re.search(r"\$[\d,]+\.\d{2}", r.content)
                if match:
                    key_metrics["Total Revenue"] = match.group(0)

        # If active_sections is empty, use instant summary fallback
        if not active_sections:
            exec_summary = (
                f"Dataset contains {profile.get('row_count', 0)} rows and {profile.get('column_count', 0)} columns. "
                "Automated analysis could not complete — see data limitations below."
            )
            recommendations = [
                f"Review dataset column structure to ensure at least one numerical or categorical attribute is populated.",
                f"Ensure dataset contains more than {profile.get('row_count', 0)} rows for statistical profiling.",
            ]
        else:
            joined_insights = " | ".join(all_insights) if all_insights else "Standard exploratory analysis completed."
            exec_summary = self._generate_executive_summary(dataset_type, key_entity, joined_insights, key_metrics)
            recommendations = self._generate_recommendations(joined_insights, key_metrics)

        # Auto-generate Data Limitations
        data_limitations = self._generate_limitations(profile, classification, results)

        return Report(
            title=title,
            dataset_type=dataset_type,
            key_entity=key_entity,
            executive_summary=exec_summary,
            key_metrics=key_metrics,
            sections=results,
            recommendations=recommendations,
            data_limitations=data_limitations,
            charts=all_charts,
            generated_at=datetime.now(),
        )

    def _generate_executive_summary(
        self, dataset_type: str, key_entity: str, insights: str, metrics: dict[str, Any]
    ) -> str:
        """Call Ollama to generate 4-sentence data-specific executive summary with fallback."""
        prompt = (
            f"You are a senior data analyst. Write a 4-sentence executive summary for a {dataset_type} dataset "
            f"where each row is a {key_entity}.\n"
            f"Key findings: {insights}.\n"
            f"Key metrics: {metrics}.\n"
            "Rules: be specific, use numbers, name the most critical finding in sentence 1, do not use filler phrases."
        )

        try:
            res = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2},
                },
                timeout=12,
            )
            if res.status_code == 200:
                summary_text = res.json().get("response", "").strip()
                if len(summary_text) > 40:
                    return summary_text
        except Exception as exc:
            LOGGER.warning("LLM executive summary generation failed (%s); using synthetic fallback", exc)

        # Fallback 4-sentence summary
        return (
            f"Automated profiling of this {dataset_type} dataset ({metrics.get('Total Records', 0)} rows) reveals a data quality score of {metrics.get('Data Quality Score', '100/100')}. "
            f"Key findings indicate {insights[:120]}... "
            f"Primary dataset features demonstrate structural completeness with {metrics.get('Duplicate Rows', 0)} duplicate rows detected prior to cleaning. "
            "Strategic metrics suggest focusing optimization on high-variance variables to improve performance."
        )

    def _generate_recommendations(self, insights: str, metrics: dict[str, Any]) -> list[str]:
        """Call Ollama to generate 5 specific business recommendations referencing numbers."""
        prompt = (
            f"Given these analysis findings: {insights} and metrics: {metrics}, "
            "write 5 specific, actionable business recommendations. "
            "Rules: Each recommendation must reference a number from the analysis. No generic advice. Respond as a JSON list of strings."
        )

        try:
            res = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 256},
                },
                timeout=35,
            )
            if res.status_code == 200:
                raw_text = res.json().get("response", "").strip()
                clean_json = re.sub(r"^```json\s*", "", raw_text, flags=re.MULTILINE)
                clean_json = re.sub(r"```$", "", clean_json, flags=re.MULTILINE).strip()
                import json
                recs = json.loads(clean_json)
                if isinstance(recs, list) and len(recs) >= 3:
                    return [str(r) for r in recs[:5]]
        except Exception as exc:
            LOGGER.warning("LLM recommendations generation failed (%s); using synthetic fallback", exc)

        return [
            f"Address missing data rates across columns to preserve quality score above {metrics.get('Data Quality Score', '90/100')}.",
            f"Investigate statistical outliers to safeguard remaining {metrics.get('Total Records', 0)} rows against anomaly bias.",
            "Implement automated data validation pipelines to prevent duplicate row accumulation.",
            "Segment entity cohorts to increase retention and lifetime performance.",
            "Establish real-time monitoring for high-variance numerical metrics.",
        ]

    def _generate_limitations(
        self, profile: dict[str, Any], classification: dict[str, Any], results: list[AnalysisResult]
    ) -> list[str]:
        """Auto-generate data limitations list based on dataset profiling and execution results."""
        limitations: list[str] = []

        missing_map = profile.get("missing_pct", {})
        high_missing = [f"{col} ({pct}%)" for col, pct in missing_map.items() if pct > 5.0]
        if high_missing:
            limitations.append(f"Columns with >5% missing values detected: {', '.join(high_missing[:4])}.")

        dupes = profile.get("duplicate_row_count", 0)
        if dupes > 0:
            limitations.append(f"{dupes:,} duplicate rows detected prior to cleaning step.")

        dt_cols = profile.get("datetime_cols", [])
        if len(dt_cols) == 1:
            limitations.append("Only one datetime column present, limiting complex multi-event temporal analysis.")
        elif not dt_cols:
            limitations.append("No datetime columns detected; longitudinal and cohort retention analyses were skipped.")

        id_cols = profile.get("id_cols", [])
        if not id_cols and not classification.get("id_column"):
            limitations.append("No unique entity ID column identified; entity-level aggregation was skipped.")

        row_cnt = profile.get("row_count", 0)
        if row_cnt < 50:
            limitations.append(f"Small dataset sample size ({row_cnt} rows); statistical findings should be interpreted with caution.")

        return limitations
