"""FastAPI entrypoint exposing uploads, queries, metrics, and pipeline wiring."""

import json
import logging
import os
import re
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

import duckdb
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from langgraph.graph import END, START, StateGraph
from neo4j.exceptions import Neo4jError
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from app.agents.analyzer import AnalyzerAgent
from app.agents.executor import ExecutorAgent
from app.agents.planner import PlannerAgent
from app.agents.state import PipelineState
from app.mcp.duckdb_tool import DuckDBTool
from app.mcp.neo4j_tool import Neo4jTool

load_dotenv()
QUERY_COUNT = Counter("datapilot_queries_total", "Total query requests")
ERROR_COUNT = Counter("datapilot_errors_total", "Total pipeline errors")
SQL_RETRY_COUNT = Counter("datapilot_sql_retries_total", "Total SQL self-correction retries")
SQL_FAILURE_COUNT = Counter("datapilot_sql_failures_total", "Total SQL failures after all retries")
STAGE_LATENCY = Histogram("datapilot_agent_latency_seconds", "Agent stage latency", ["stage"])
REPORT_GEN_LATENCY = Histogram("datapilot_report_generation_latency_seconds", "Dataset report generation latency seconds")


class QueryRequest(BaseModel):
    """Request body for natural-language queries."""

    question: str
    history: list[dict[str, str]] = Field(default_factory=list)


class DataPilotApp:
    """Application dependencies and compiled LangGraph pipeline."""

    def __init__(self) -> None:
        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        duckdb_tool = DuckDBTool(os.getenv("DUCKDB_PATH", "./data/datapilot.duckdb"))
        neo4j_tool = Neo4jTool(os.getenv("NEO4J_URI", "bolt://localhost:7687"), os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "datapilot"))
        planner = PlannerAgent(duckdb_tool, model, base_url)
        executor = ExecutorAgent(duckdb_tool, neo4j_tool, model=model, base_url=base_url)
        analyzer = AnalyzerAgent(model, base_url, os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/datapilot-anomaly"))

        def planner_node(state: PipelineState) -> PipelineState:
            with STAGE_LATENCY.labels("planner").time():
                return planner.run(state)

        def executor_node(state: PipelineState) -> PipelineState:
            with STAGE_LATENCY.labels("executor").time():
                return executor.run(state)

        def analyzer_node(state: PipelineState) -> PipelineState:
            with STAGE_LATENCY.labels("analyzer").time():
                return analyzer.run(state)

        graph = StateGraph(PipelineState)
        graph.add_node("planner", planner_node)
        graph.add_node("executor", executor_node)
        graph.add_node("analyzer", analyzer_node)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "executor")
        graph.add_edge("executor", "analyzer")
        graph.add_edge("analyzer", END)
        self.pipeline = graph.compile()
        self.duckdb_tool = duckdb_tool
        self.neo4j_tool = neo4j_tool


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Create and dispose shared database and pipeline dependencies."""
    application.state.datapilot = DataPilotApp()
    yield
    application.state.datapilot.neo4j_tool.close()


app = FastAPI(title="DataPilot AI", version="0.1.0", lifespan=lifespan)


@app.post("/query")
def query(request: QueryRequest) -> dict[str, Any]:
    """Run the Planner -> Executor -> Analyzer pipeline."""
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question must not be empty.")
    QUERY_COUNT.inc()
    try:
        return dict(
            app.state.datapilot.pipeline.invoke(
                {"question": request.question, "history": request.history[-8:]}
            )
        )
    except (ValueError, json.JSONDecodeError, duckdb.Error, Neo4jError, requests.RequestException) as exc:
        ERROR_COUNT.inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """Seed DuckDB from CSV, JSON, Excel, or Parquet upload with databroom cleaning."""
    from app.services.cleaning import sanitize_table_name

    suffix = Path(file.filename or "").suffix.lower()
    supported_suffixes = {".csv", ".json", ".xlsx", ".xls", ".parquet"}
    if suffix not in supported_suffixes:
        raise HTTPException(status_code=415, detail=f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(supported_suffixes))}")

    existing_tables = app.state.datapilot.duckdb_tool.list_tables()
    table_name = sanitize_table_name(Path(file.filename or "uploaded").stem, existing_tables)

    # Persist the untouched raw file on disk
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_disk_path = raw_dir / f"{table_name}_raw{suffix}"
    
    file_bytes = await file.read()
    with open(raw_disk_path, "wb") as f:
        f.write(file_bytes)

    # Load and clean via AutoAnalysisPipeline into DuckDB and generate Report
    try:
        import pandas as pd
        from app.analysis.pipeline import AutoAnalysisPipeline

        if suffix == ".csv":
            df_raw = pd.read_csv(raw_disk_path)
        elif suffix in [".xlsx", ".xls"]:
            df_raw = pd.read_excel(raw_disk_path)
        elif suffix == ".json":
            df_raw = pd.read_json(raw_disk_path)
        elif suffix == ".parquet":
            df_raw = pd.read_parquet(raw_disk_path)
        else:
            df_raw = pd.read_csv(raw_disk_path)

        pipeline = AutoAnalysisPipeline()
        report = pipeline.run(df_raw, table_name=table_name)
        report_dict = report.to_dict()
        app.state.datapilot.duckdb_tool.last_dataset_report = report_dict

        # Count clean rows loaded
        with duckdb.connect(app.state.datapilot.duckdb_tool.database_path) as conn:
            rows = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

    except Exception as exc:
        LOGGER.error("Upload & auto-analysis pipeline failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to process and analyze dataset: {exc}") from exc

    raw_qs = report_dict.get("key_metrics", {}).get("Data Quality Score", 100)
    if isinstance(raw_qs, str) and "/" in raw_qs:
        try:
            clean_qs = float(raw_qs.split("/")[0])
        except ValueError:
            clean_qs = 100.0
    else:
        try:
            clean_qs = float(raw_qs)
        except (ValueError, TypeError):
            clean_qs = 100.0

    dupes_str = report_dict.get("key_metrics", {}).get("Duplicate Rows", "0")
    try:
        dupes_cnt = int(str(dupes_str).replace(",", ""))
    except ValueError:
        dupes_cnt = 0

    return {
        "table": table_name,
        "rows": rows,
        "cleaning": {
            "clean_score": clean_qs,
            "quality_score": clean_qs,
            "rows_after": rows,
            "duplicates_removed": dupes_cnt,
        },
        "report": report_dict,
    }


@app.get("/datasets")
def list_datasets() -> list[dict[str, Any]]:
    """Return all loaded user datasets with report summaries."""
    return app.state.datapilot.duckdb_tool.get_all_dataset_summaries()


@app.get("/datasets/{name}/report")
def get_dataset_report(name: str, version: int | None = None) -> dict[str, Any]:
    """Return JSON dataset report for a table."""
    with REPORT_GEN_LATENCY.time():
        report = app.state.datapilot.duckdb_tool.get_dataset_report(name, version=version)
        if not report:
            raise HTTPException(status_code=404, detail=f"No report found for dataset '{name}'.")
        return report


@app.get("/datasets/{name}/report.html", response_class=HTMLResponse)
def get_dataset_report_html(name: str, version: int | None = None) -> HTMLResponse:
    """Return self-contained HTML report with inline CSS."""
    from app.services.report_renderer import render_report_html

    with REPORT_GEN_LATENCY.time():
        report = app.state.datapilot.duckdb_tool.get_dataset_report(name, version=version)
        if not report:
            raise HTTPException(status_code=404, detail=f"No report found for dataset '{name}'.")
        html_content = render_report_html(report)
        return HTMLResponse(content=html_content)


@app.post("/reset")
def reset() -> dict[str, str]:
    """Clear all user tables from DuckDB and reset database state."""
    app.state.datapilot.duckdb_tool.reset_database()
    return {"message": "Database and active tables successfully reset."}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> PlainTextResponse:
    """Expose Prometheus metrics."""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
