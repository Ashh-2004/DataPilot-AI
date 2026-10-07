"""Executor agent: executes plans, runs self-correction loop, and applies SQL guardrails."""

from datetime import date
import logging
import os
import re
from pathlib import Path
from typing import Any

import requests
import yaml
from langchain_ollama import ChatOllama

from app.mcp.duckdb_tool import DuckDBTool
from app.mcp.neo4j_tool import Neo4jTool
from app.services.sql_guardrails import SQLValidationError, validate_sql

LOGGER = logging.getLogger(__name__)
MAX_RETRIES = 3


class ExecutorAgent:
    """Executes SQL queries with EXPLAIN validation, self-correction, and graceful fallbacks."""

    def __init__(
        self,
        duckdb_tool: DuckDBTool,
        neo4j_tool: Neo4jTool,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.duckdb_tool = duckdb_tool
        self.neo4j_tool = neo4j_tool
        self.model = model
        self.base_url = base_url
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0,
            num_ctx=3072,
        )
        self.few_shot_examples = self._load_few_shot_examples()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
        self.sql_backend = os.getenv("SQL_BACKEND", "ollama").lower()

    @staticmethod
    def _load_few_shot_examples() -> str:
        """Load few-shot examples from app/prompts/examples.yaml."""
        examples_path = Path(__file__).resolve().parent.parent / "prompts" / "examples.yaml"
        if not examples_path.exists():
            return ""
        try:
            with open(examples_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            examples = data.get("examples", [])
            lines = ["Here are examples of questions and their correct DuckDB SQL queries:"]
            for ex in examples[:8]:
                q = ex.get("question", "")
                sql = ex.get("sql", "").strip()
                lines.append(f"Question: {q}\nSQL: {sql}\n")
            return "\n".join(lines)
        except Exception:
            return ""

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the query plan with self-correction retry loop."""
        from app.api.main import SQL_FAILURE_COUNT, SQL_RETRY_COUNT

        plan = state.get("plan", {})
        question = state.get("question", "")
        schema = state.get("schema_context", "") or self.duckdb_tool.schema_context()
        initial_sql = str(plan.get("sql_hint", "")).strip()

        sql_attempts: list[dict[str, Any]] = []
        current_sql = initial_sql
        results: list[dict[str, Any]] = []
        last_error: str | None = None

        # Self-correction loop: attempt initial SQL, then up to MAX_RETRIES
        for attempt in range(MAX_RETRIES + 1):
            if not current_sql:
                last_error = "No SQL statement generated"
                sql_attempts.append({"attempt": attempt, "sql": "", "error": last_error})
                current_sql = self._generate_sql(question, schema, attempt=attempt, prev_error=last_error)
                continue

            # Strip markdown fences
            clean_sql = self._clean_sql_string(current_sql)

            # Step 1: Pre-flight EXPLAIN check & validation
            explain_err = self.duckdb_tool.explain_query(clean_sql)
            if explain_err:
                LOGGER.warning("Attempt %d SQL failed EXPLAIN: %s (query: %s)", attempt + 1, explain_err, clean_sql)
                sql_attempts.append({"attempt": attempt + 1, "sql": clean_sql, "error": explain_err})
                last_error = explain_err
                if attempt < MAX_RETRIES:
                    SQL_RETRY_COUNT.inc()
                    current_sql = self._generate_sql(question, schema, attempt=attempt + 1, prev_error=explain_err)
                continue

            # Step 2: Execution
            try:
                results = self.duckdb_tool.execute_read_only(clean_sql)
                sql_attempts.append({"attempt": attempt + 1, "sql": clean_sql, "error": None, "rows": len(results)})
                
                # Check for suspicious empty results on answerable questions
                if len(results) == 0 and attempt < MAX_RETRIES and self._seems_suspiciously_empty(clean_sql):
                    LOGGER.info("Attempt %d returned 0 rows suspiciously; attempting query relaxation", attempt + 1)
                    SQL_RETRY_COUNT.inc()
                    current_sql = self._generate_sql(
                        question,
                        schema,
                        attempt=attempt + 1,
                        prev_error="Query returned 0 rows. The filter conditions may be too strict or case-sensitive.",
                    )
                    continue

                # Query succeeded!
                LOGGER.info("Query succeeded on attempt %d: %s (returned %d rows)", attempt + 1, clean_sql, len(results))
                self.neo4j_tool.record_query(question, {"sql_hint": clean_sql, "intent": plan.get("intent", "")})
                columns = list(results[0].keys()) if results else []
                return {
                    "results": results,
                    "columns": columns,
                    "final_sql": clean_sql,
                    "sql_attempts": sql_attempts,
                    "retry_count": attempt,
                    "error": None,
                }
            except Exception as exc:
                last_error = str(exc)
                LOGGER.warning("Attempt %d SQL execution error: %s", attempt + 1, exc)
                sql_attempts.append({"attempt": attempt + 1, "sql": clean_sql, "error": last_error})
                if attempt < MAX_RETRIES:
                    SQL_RETRY_COUNT.inc()
                    current_sql = self._generate_sql(question, schema, attempt=attempt + 1, prev_error=last_error)

        # All retries exhausted: graceful fallback
        SQL_FAILURE_COUNT.inc()
        LOGGER.error("All %d SQL attempts failed for question %r: %s", MAX_RETRIES + 1, question, last_error)
        fallback = self._build_graceful_fallback(question, last_error, sql_attempts)
        return {
            "results": [],
            "columns": [],
            "final_sql": None,
            "sql_attempts": sql_attempts,
            "retry_count": MAX_RETRIES,
            "error": last_error,
            "answer": fallback["answer"],
            "key_insights": fallback["key_insights"],
            "limitations": fallback["limitations"],
            "follow_up_questions": fallback["follow_up_questions"],
        }

    def _generate_sql(self, question: str, schema: str, attempt: int, prev_error: str | None) -> str:
        """Generate or fix SQL using Ollama or optional Gemini backend with different retry strategies."""
        today_str = date.today().isoformat()
        user_tables = self.duckdb_tool.list_user_tables()
        tables_str = ", ".join([f'"{t}"' for t in user_tables]) if user_tables else "no table loaded"
        
        # Strategy progression across retries
        if attempt == 1:
            strategy = "STRATEGY: FIX THE SPECIFIC ERROR. Adjust column names, types, or DuckDB syntax."
        elif attempt == 2:
            strategy = "STRATEGY: SIMPLIFY THE QUERY. Remove complex subqueries or overly restrictive filters."
        else:
            strategy = "STRATEGY: RE-PLAN FROM SCRATCH. Write the simplest possible SELECT query that answers the question."

        prompt = (
            "You are an expert DuckDB SQL engineer.\n"
            f"Available Database Tables: {tables_str}\n"
            f"Database Schema:\n{schema}\n\n"
            f"{self.few_shot_examples}\n\n"
            f"Current Date: {today_str}\n"
            f"User Question: {question}\n\n"
            f"Previous Attempt Failed With Error:\n{prev_error}\n\n"
            f"{strategy}\n\n"
            "RULES:\n"
            "1. Output ONLY the raw SQL query. No markdown backticks, no quotes around the whole block, no conversational text.\n"
            f"2. You MUST ONLY query tables that exist in the database: {tables_str}.\n"
            "3. Dialect: DuckDB only. Quote identifiers with double quotes: \"table_name\".\"column_name\".\n"
            "4. Relative dates: use DuckDB current_date, date_trunc('month', current_date), INTERVAL '1 month'.\n"
            "5. Case-insensitivity: Use ILIKE or LOWER() when filtering text if exact case might differ.\n"
            "6. Single statement only.\n\n"
            "SQL Query:"
        )

        # Use Gemini if configured and available
        if self.sql_backend == "gemini" and self.gemini_api_key:
            try:
                gemini_sql = self._call_gemini(prompt)
                if gemini_sql:
                    return gemini_sql
            except Exception as exc:
                LOGGER.warning("Gemini SQL generation failed: %s; falling back to Ollama", exc)

        # Default Ollama generation
        try:
            response = self.llm.invoke(prompt)
            content = response.content if isinstance(response.content, str) else str(response.content)
            return self._extract_sql(content)
        except Exception as exc:
            LOGGER.warning("Ollama SQL retry generation failed: %s", exc)
            return ""

    def _call_gemini(self, prompt: str) -> str:
        """Optional stronger backend: direct Google Gemini REST API."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 300},
        }
        res = requests.post(url, json=payload, timeout=20)
        res.raise_for_status()
        data = res.json()
        candidates = data.get("candidates", [])
        if candidates:
            text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return self._extract_sql(text)
        return ""

    @staticmethod
    def _extract_sql(text: str) -> str:
        """Extract pure SQL query from model response."""
        cleaned = text.strip()
        # Find markdown sql code fence
        match = re.search(r"```(?:sql)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # If text contains SELECT or WITH
        select_match = re.search(r"\b(SELECT|WITH)\b.*", cleaned, re.DOTALL | re.IGNORECASE)
        if select_match:
            return select_match.group(0).strip()
        return cleaned

    @staticmethod
    def _clean_sql_string(sql: str) -> str:
        """Strip markdown code fences and extraneous text."""
        cleaned = re.sub(r"(?is)```(?:sql)?", "", sql).replace("```", "").strip()
        # Strip trailing semicolon
        if cleaned.endswith(";"):
            cleaned = cleaned[:-1].rstrip()
        return cleaned

    @staticmethod
    def _seems_suspiciously_empty(sql: str) -> bool:
        """Check if query had an equality filter that might have yielded 0 rows due to case-sensitivity."""
        return "=" in sql or "LIKE" in sql.upper()

    def _build_graceful_fallback(
        self,
        question: str,
        last_error: str | None,
        attempts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Produce an informative explanation and helpful alternatives when SQL cannot execute."""
        user_tables = self.duckdb_tool.list_user_tables()
        all_cols = self.duckdb_tool.get_all_column_names()
        sample_cols = ", ".join(all_cols[:8])

        is_unanswerable = any(
            token in question.lower()
            for token in ("satisfaction", "profit margin", "churn", "net promoter", "inventory")
        )
        if is_unanswerable:
            answer = (
                f"I cannot answer '{question}' because the required metrics or columns do not exist "
                f"in the dataset. Available tables ({', '.join(user_tables)}) contain: {sample_cols}."
            )
        else:
            answer = (
                f"I was unable to retrieve data for '{question}'. I tried {len(attempts)} query variations, "
                f"but encountered database error: '{last_error}'.\n"
                f"Available columns in the dataset include: {sample_cols}."
            )

        return {
            "answer": answer,
            "key_insights": ["The requested data or query could not be resolved against the database schema."],
            "limitations": [f"Database error: {last_error}" if last_error else "Query execution failed."],
            "follow_up_questions": [
                f"Show summary statistics for {user_tables[0]}" if user_tables else "Describe the dataset",
                f"What are the top 5 records in {user_tables[0]}?" if user_tables else "Show table records",
                "What columns are available in this table?",
            ],
        }
