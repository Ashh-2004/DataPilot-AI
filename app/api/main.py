"""FastAPI entrypoint exposing uploads, queries, metrics, and pipeline wiring."""

import json
import os
import re
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import duckdb
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
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
STAGE_LATENCY = Histogram("datapilot_agent_latency_seconds", "Agent stage latency", ["stage"])


class QueryRequest(BaseModel):
    """Request body for natural-language queries."""

    question: str
    history: list[dict[str, str]] = Field(default_factory=list)


class DataPilotApp:
    """Application dependencies and compiled LangGraph pipeline."""

    def __init__(self) -> None:
        duckdb_tool = DuckDBTool(os.getenv("DUCKDB_PATH", "/app/data/datapilot.duckdb"))
        neo4j_tool = Neo4jTool(os.getenv("NEO4J_URI", "bolt://neo4j:7687"), os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "datapilot"))
        planner = PlannerAgent(duckdb_tool, os.getenv("OLLAMA_MODEL", "llama3.2"), os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"))
        executor = ExecutorAgent(duckdb_tool, neo4j_tool)
        analyzer = AnalyzerAgent(os.getenv("OLLAMA_MODEL", "llama3.2"), os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"), os.getenv("N8N_WEBHOOK_URL", "http://n8n:5678/webhook/datapilot-anomaly"))

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
    """Seed DuckDB from one CSV or JSON upload."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".csv", ".json"}:
        raise HTTPException(status_code=415, detail="Only CSV and JSON files are supported.")
    table_name = re.sub(r"[^A-Za-z0-9_]", "_", Path(file.filename or "uploaded").stem)
    if not table_name or table_name[0].isdigit():
        table_name = f"uploaded_{table_name}"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
        temporary.write(await file.read())
        temporary_path = temporary.name
    try:
        rows = app.state.datapilot.duckdb_tool.load_file(temporary_path, table_name)
    finally:
        Path(temporary_path).unlink(missing_ok=True)
    return {"table": table_name, "rows": rows}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> PlainTextResponse:
    """Expose Prometheus metrics."""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
