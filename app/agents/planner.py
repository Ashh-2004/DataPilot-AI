"""Planner agent: converts natural language into a validated structured execution plan."""

from datetime import date
import json
import logging
import os
import re
from typing import Any

from langchain_ollama import ChatOllama
from rapidfuzz import fuzz, process

from app.mcp.duckdb_tool import DuckDBTool
from app.services.query_rewriter import rewrite_followup

LOGGER = logging.getLogger(__name__)


class PlannerAgent:
    """Uses Ollama to produce a constrained, schema-validated plan for the executor."""

    def __init__(self, duckdb_tool: DuckDBTool, model: str, base_url: str) -> None:
        self.duckdb_tool = duckdb_tool
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0,
            num_ctx=3072,
            format="json",  # Force Ollama JSON output
        )
        self.rewriter_llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0,
            num_ctx=1024,
        )

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Create a plan containing intent, tables, columns, filters, and SQL."""
        schema = self.duckdb_tool.schema_context()
        if schema == "No tables are loaded.":
            raise ValueError("Upload a dataset before asking a question.")

        question = state.get("question", "").strip()
        history = state.get("history", [])

        # Step 1: Follow-up rewriting if conversation history exists
        rewritten_question = question
        if history:
            rewritten_question = rewrite_followup(question, history, self.rewriter_llm)

        # Overview shortcut
        if self._is_overview_question(rewritten_question):
            overview = self.duckdb_tool.dataset_overview()
            tables = self.duckdb_tool.list_user_tables()
            table = tables[0] if tables else "uploaded_table"
            return {
                "question": rewritten_question,
                "original_question": question,
                "schema_context": schema,
                "dataset_overview": overview,
                "plan": {
                    "intent": "dataset_overview",
                    "tables": [table],
                    "columns": ["*"],
                    "sql_hint": f'SELECT * FROM "{table}" LIMIT 25',
                },
            }

        # Build schema and valid column catalog for validation
        user_tables = self.duckdb_tool.list_user_tables()
        all_columns = self.duckdb_tool.get_all_column_names()
        today_str = date.today().isoformat()

        # Load dataset_profile, dataset_classification, and data_sample from Redis / memory
        profile = None
        classification = None
        sample = None

        try:
            import redis
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", 6379))
            r = redis.Redis(host=redis_host, port=redis_port, db=0, socket_connect_timeout=2)
            p_str = r.get("dataset_profile")
            c_str = r.get("dataset_classification")
            s_str = r.get("data_sample")

            if p_str:
                profile = json.loads(p_str)
            if c_str:
                classification = json.loads(c_str)
            if s_str:
                sample = json.loads(s_str)
        except Exception as exc:
            LOGGER.info("Redis lookup in Planner: %s", exc)

        if not profile or not classification:
            from app.analysis.pipeline import MEMORY_CACHE
            profile = MEMORY_CACHE.get("dataset_profile")
            classification = MEMORY_CACHE.get("dataset_classification")
            sample = MEMORY_CACHE.get("data_sample")

        if not profile or not classification:
            return {
                "question": rewritten_question,
                "original_question": question,
                "schema_context": schema,
                "answer": "Please upload a dataset first before asking questions.",
                "plan": {
                    "intent": "missing_dataset",
                    "tables": [user_tables[0]] if user_tables else [],
                    "columns": [],
                    "sql_hint": f'SELECT \'Please upload a dataset first before asking questions.\' AS message',
                },
            }

        col_types = profile.get("column_types", profile.get("columns", {}))
        if isinstance(col_types, dict) and any(isinstance(v, dict) for v in col_types.values()):
            flat_types = {k: v.get("type", "unknown") if isinstance(v, dict) else str(v) for k, v in col_types.items()}
        else:
            flat_types = col_types

        system_context = f"""
You are a data analyst assistant. The user has uploaded a dataset.
Dataset type: {classification.get('dataset_type', 'generic')}
Each row represents: {classification.get('key_entity', 'record')}
Columns: {list(flat_types.keys()) if isinstance(flat_types, dict) else list(profile.keys())}
Column types: {flat_types}
Sample rows: {sample}
Key stats: {profile.get('numeric_stats', {})}
Total rows: {profile.get('row_count', 0)}

When answering questions:
- Always reference actual column names from this dataset
- Use real numbers from the stats above
- Never write code unless the user explicitly asks for it
- Answer as if you already know this dataset deeply
"""

        prompt_text = (
            f"{system_context}\n\n"
            "You are a text-to-SQL planner and query generator for DuckDB.\n"
            f"Today's date is: {today_str}. Use this for resolving relative dates ('this month', 'last quarter', 'this year').\n\n"
            f"Available Database Tables and Schema:\n{schema}\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Output ONLY a valid JSON object. No explanation, no markdown backticks.\n"
            f"2. You MUST only query tables that exist in the schema: {', '.join(user_tables)}.\n"
            "3. Required JSON keys:\n"
            "   - 'intent': brief string description of intent (e.g. 'sum_revenue', 'top_products')\n"
            "   - 'tables': array of table names referenced (must be in schema)\n"
            "   - 'columns': array of column names needed\n"
            "   - 'filters': array of filter conditions\n"
            "   - 'sql_hint': a single COMPLETE, valid DuckDB SQL query starting with SELECT or WITH\n"
            "4. SQL Rules:\n"
            "   - 'sql_hint' MUST be a full executable SELECT query (e.g. SELECT * FROM \"table\" WHERE ...). NEVER output just a WHERE clause or fragment.\n"
            "   - Dialect: DuckDB only. Quote all table and column names with double quotes.\n"
            "   - For relative dates, use DuckDB functions: date_trunc('month', current_date), INTERVAL, etc.\n"
            "   - Always include appropriate GROUP BY and ORDER BY clauses.\n"
            "   - Do NOT use parameters or question marks.\n\n"
            f"User Question: {rewritten_question}\n"
            "JSON Response:"
        )

        try:
            response = self.llm.invoke(prompt_text)
            content = response.content if isinstance(response.content, str) else str(response.content)
            plan = self._extract_json(content)
        except Exception as exc:
            LOGGER.warning("LLM planner invocation failed: %s; using heuristic plan", exc)
            plan = self._heuristic_fallback_plan(rewritten_question, user_tables, all_columns)

        # Validate and repair plan against schema
        plan = self._validate_and_repair_plan(plan, user_tables, all_columns)

        return {
            "question": rewritten_question,
            "original_question": question,
            "schema_context": schema,
            "plan": plan,
        }

    @staticmethod
    def _extract_json(content: str) -> dict[str, Any]:
        """Extract and parse JSON object from model output."""
        cleaned = content.strip()
        # Remove code fences if present
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise ValueError(f"No JSON object found in response: {content[:100]}")

    def _validate_and_repair_plan(
        self,
        plan: dict[str, Any],
        user_tables: list[str],
        all_columns: list[str],
    ) -> dict[str, Any]:
        """Validate referenced tables and columns; apply fuzzy matching if mismatched."""
        if not isinstance(plan, dict):
            plan = {}

        primary_table = user_tables[0] if user_tables else "data"

        # Ensure required keys exist
        if "intent" not in plan:
            plan["intent"] = "query"
        if "tables" not in plan or not isinstance(plan["tables"], list):
            plan["tables"] = [primary_table]

        # Check tables match actual tables
        valid_tables = []
        for t in plan["tables"]:
            if t in user_tables:
                valid_tables.append(t)
            elif user_tables:
                match = process.extractOne(t, user_tables, scorer=fuzz.ratio)
                if match and match[1] >= 60:
                    valid_tables.append(match[0])
                    if "sql_hint" in plan and plan["sql_hint"]:
                        plan["sql_hint"] = re.sub(rf'\b"{t}"\b|\b{t}\b', f'"{match[0]}"', str(plan["sql_hint"]))
        plan["tables"] = valid_tables or [primary_table]

        sql = str(plan.get("sql_hint", "")).strip()

        # 1. Ensure sql_hint starts with SELECT or WITH
        clean_sql = re.sub(r"(?is)```(?:sql)?", "", sql).replace("```", "").strip()
        if not (clean_sql.upper().startswith("SELECT") or clean_sql.upper().startswith("WITH")):
            if "=" in clean_sql or "ILIKE" in clean_sql.upper() or "LIKE" in clean_sql.upper():
                sql = f'SELECT * FROM "{primary_table}" WHERE {clean_sql} LIMIT 500'
            else:
                sql = f'SELECT * FROM "{primary_table}" LIMIT 500'

        # 2. Ensure table names in SQL match actual existing user tables
        for t in re.findall(r'FROM\s+["`]?([A-Za-z0-9_]+)["`]?', sql, re.IGNORECASE):
            if t not in user_tables and user_tables:
                sql = re.sub(rf'\b"{t}"\b|\b{t}\b', f'"{primary_table}"', sql)

        # 3. Column fuzzy matching
        for t in plan["tables"]:
            mapping = self.duckdb_tool.get_column_mapping(t)
            for raw_col, clean_col in mapping.items():
                if raw_col != clean_col and raw_col in sql:
                    sql = sql.replace(f'"{raw_col}"', f'"{clean_col}"').replace(raw_col, f'"{clean_col}"')

        plan["sql_hint"] = sql
        return plan

    @staticmethod
    def _heuristic_fallback_plan(question: str, user_tables: list[str], all_columns: list[str]) -> dict[str, Any]:
        """Build a safe basic plan when planner LLM is unresponsive."""
        table = user_tables[0] if user_tables else "data"
        return {
            "intent": "select_fallback",
            "tables": [table],
            "columns": ["*"],
            "sql_hint": f'SELECT * FROM "{table}" LIMIT 25',
        }

    @staticmethod
    def _is_overview_question(question: str) -> bool:
        """Route broad walkthrough and dataset identification requests without complex SQL queries."""
        normalized = question.casefold().strip()
        markers = (
            "walk me through",
            "give me an overview",
            "dataset overview",
            "describe the dataset",
            "explain this dataset",
            "explain the dataset",
            "tell me about this dataset",
            "tell me about this data",
            "what is in this dataset",
            "what is in the dataset",
            "summarize the dataset",
            "explore this dataset",
            "what is this dataset",
            "what dataset is this",
            "what is this data",
            "what data is this",
            "what is this dataset about",
            "what is this data about",
            "what is the dataset about",
            "what is the data about",
            "what data did i upload",
            "what did i upload",
            "what data do we have",
            "about this dataset",
            "about the dataset",
            "overview of dataset",
            "overview of the data",
            "describe dataset",
        )
        return any(marker in normalized for marker in markers)
