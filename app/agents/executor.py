"""Executor agent: runs the planner's read-only query through MCP tools."""

from typing import Any

from app.mcp.duckdb_tool import DuckDBTool
from app.mcp.neo4j_tool import Neo4jTool


class ExecutorAgent:
    """Executes plans without making direct database calls."""

    def __init__(self, duckdb_tool: DuckDBTool, neo4j_tool: Neo4jTool) -> None:
        self.duckdb_tool = duckdb_tool
        self.neo4j_tool = neo4j_tool

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Query DuckDB and persist query memory."""
        plan = state["plan"]
        sql = str(plan.get("sql_hint", "")).strip()
        if not sql:
            raise ValueError("Planner did not provide SQL.")
        if "?" in sql or "$" in sql:
            raise ValueError("Planner returned SQL with unsupported parameters.")
        results = self.duckdb_tool.execute_read_only(sql)
        self.neo4j_tool.record_query(state["question"], plan)
        return {"results": results, "columns": list(results[0].keys()) if results else []}
