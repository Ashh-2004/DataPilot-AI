"""Typed state shared by the three DataPilot LangGraph nodes."""

from typing import Any, TypedDict


class PipelineState(TypedDict, total=False):
    """State passed explicitly between pipeline agents."""

    question: str
    history: list[dict[str, str]]
    dataset_overview: dict[str, Any]
    plan: dict[str, Any]
    results: list[dict[str, Any]]
    columns: list[str]
    answer: str
    key_insights: list[str]
    limitations: list[str]
    follow_up_questions: list[str]
    insight: str
    chart: dict[str, Any] | None
    anomaly: bool
    anomaly_details: dict[str, Any] | None
    anomaly_result: dict[str, Any]
    error: str | None
