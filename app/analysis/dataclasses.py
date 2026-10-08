"""Dataclasses for universal auto-analysis pipeline, task specification, and report generation."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import pandas as pd


@dataclass
class ChartSpec:
    """Specification for Plotly chart rendering."""

    chart_type: str  # "line" | "bar" | "histogram" | "scatter" | "heatmap" | "box" | "pie" | "area"
    title: str
    x: str
    y: str | list[str]
    color: str | None = None
    data: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class AnalysisTask:
    """Task specification for dynamic analysis planning."""

    name: str
    description: str
    required_columns: list[str]
    analysis_fn: Callable[..., "AnalysisResult"]


@dataclass
class AnalysisResult:
    """Outcome of a single analysis task execution."""

    section_title: str
    content: str  # Plain English findings
    table: pd.DataFrame | None = None
    insight: str = ""  # Single most important finding
    charts: list[ChartSpec] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None


@dataclass
class Report:
    """Final assembled executive dataset report."""

    title: str
    dataset_type: str
    key_entity: str
    executive_summary: str
    key_metrics: dict[str, Any]
    sections: list[AnalysisResult]
    recommendations: list[str]
    data_limitations: list[str]
    charts: list[ChartSpec]
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert Report to JSON-serializable dictionary."""
        return {
            "title": self.title,
            "dataset_type": self.dataset_type,
            "key_entity": self.key_entity,
            "executive_summary": self.executive_summary,
            "key_metrics": self.key_metrics,
            "sections": [
                {
                    "section_title": s.section_title,
                    "content": s.content,
                    "table": s.table.to_dict(orient="records") if s.table is not None else None,
                    "insight": s.insight,
                    "skipped": s.skipped,
                    "skip_reason": s.skip_reason,
                }
                for s in self.sections
            ],
            "recommendations": self.recommendations,
            "data_limitations": self.data_limitations,
            "charts": [
                {
                    "chart_type": c.chart_type,
                    "title": c.title,
                    "x": c.x,
                    "y": c.y,
                    "color": c.color,
                    "data": c.data.to_dict(orient="records") if isinstance(c.data, pd.DataFrame) else [],
                }
                for c in self.charts
            ],
            "generated_at": self.generated_at.isoformat(),
        }
