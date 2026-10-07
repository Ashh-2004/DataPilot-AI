"""SQL guardrails: validation, sanitization, and safety enforcement.

Uses sqlglot for parsing and AST inspection rather than fragile regexes.
Enforces SELECT/WITH only, single statement, no DDL/DML/COPY/ATTACH, LIMIT cap, and query timeouts.
"""

import logging
import re
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_ROW_LIMIT = 10_000
MAX_ROW_LIMIT = 50_000
QUERY_TIMEOUT_SECONDS = 30

# Forbidden statement types (AST node classes)
_FORBIDDEN_TYPES = {
    exp.Create, exp.Drop, exp.Insert, exp.Update, exp.Delete,
    exp.Alter, exp.AlterColumn,
}

# Forbidden function / command names (case-insensitive)
_FORBIDDEN_NAMES = {
    "copy", "attach", "install", "load", "pragma", "export",
    "import", "checkpoint", "vacuum", "call",
}


class SQLValidationError(ValueError):
    """Raised when a SQL statement violates guardrails."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_sql(sql: str) -> str:
    """Validate and normalise a SQL string for safe read-only execution.

    Returns the cleaned SQL on success.
    Raises SQLValidationError on any violation.
    """
    sql = _strip_markdown(sql)
    if not sql:
        raise SQLValidationError("Empty SQL statement.")

    # --- Quick regex pre-filter for obviously dangerous patterns -----------
    upper = sql.upper()
    for keyword in ("COPY", "ATTACH", "INSTALL", "PRAGMA", "EXPORT", "IMPORT"):
        if re.search(rf"\b{keyword}\b", upper):
            raise SQLValidationError(f"Forbidden SQL keyword: {keyword}")

    # --- Parse with sqlglot ------------------------------------------------
    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except ParseError as exc:
        raise SQLValidationError(f"SQL parse error: {exc}") from exc

    if not statements:
        raise SQLValidationError("No SQL statements found.")
    if len(statements) > 1:
        raise SQLValidationError("Only a single SQL statement is allowed.")

    statement = statements[0]
    if statement is None:
        raise SQLValidationError("Failed to parse SQL statement.")

    # --- Check statement type ----------------------------------------------
    if not isinstance(statement, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        # Allow CTEs (WITH ... SELECT)
        if not (hasattr(statement, "find") and statement.find(exp.Select)):
            raise SQLValidationError(
                f"Only SELECT/WITH queries are allowed, got {type(statement).__name__}."
            )

    # --- Walk AST for forbidden nodes --------------------------------------
    for node in statement.walk():
        if type(node) in _FORBIDDEN_TYPES:
            raise SQLValidationError(f"Forbidden SQL operation: {type(node).__name__}.")
        # Check function/command names
        if isinstance(node, (exp.Anonymous, exp.Func)):
            name = getattr(node, "name", "") or ""
            if name.lower() in _FORBIDDEN_NAMES:
                raise SQLValidationError(f"Forbidden SQL function/command: {name}.")

    # --- Ensure a LIMIT exists; add one if missing -------------------------
    cleaned_sql = _ensure_limit(statement)

    return cleaned_sql


def explain_sql(sql: str, connection: Any) -> str | None:
    """Run EXPLAIN on a SQL statement and return the error message, or None on success."""
    try:
        connection.execute(f"EXPLAIN {sql}")
        return None
    except Exception as exc:
        return str(exc)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_markdown(sql: str) -> str:
    """Remove markdown code fences and trailing semicolons."""
    cleaned = re.sub(r"(?is)```(?:sql)?", "", sql).replace("```", "").strip()
    # Remove single trailing semicolon (leave SQL intact)
    if cleaned.endswith(";"):
        if cleaned.count(";") == 1 or not cleaned[:-1].rstrip().endswith(";"):
            cleaned = cleaned[:-1].rstrip()
    return cleaned


def _ensure_limit(statement: exp.Expression) -> str:
    """If the outermost SELECT has no LIMIT, add one."""
    # Find the outermost select
    select = statement if isinstance(statement, exp.Select) else statement.find(exp.Select)
    if select is None:
        return statement.sql(dialect="duckdb")

    # Check if LIMIT already present
    limit_node = statement.find(exp.Limit)
    if limit_node is not None:
        # Cap excessive limits
        limit_val = limit_node.expression
        if isinstance(limit_val, exp.Literal) and limit_val.is_int:
            val = int(limit_val.this)
            if val > MAX_ROW_LIMIT:
                limit_val.set("this", str(MAX_ROW_LIMIT))
        return statement.sql(dialect="duckdb")

    # Add a default limit
    try:
        limited = statement.copy()
        # Use sqlglot to add limit
        outer = limited if isinstance(limited, exp.Select) else limited.find(exp.Select)
        if outer is not None:
            result_sql = limited.sql(dialect="duckdb") + f" LIMIT {DEFAULT_ROW_LIMIT}"
            return result_sql
    except Exception:
        pass

    return statement.sql(dialect="duckdb") + f" LIMIT {DEFAULT_ROW_LIMIT}"

