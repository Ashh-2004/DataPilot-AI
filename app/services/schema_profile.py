"""Rich schema profiling for uploaded tables.

Builds per-column statistics (null%, distinct count, min/max, sample values)
injected into the LLM prompts so the model knows exactly what data exists.
"""

import logging
from datetime import date, datetime
from typing import Any

import duckdb
import pandas as pd

LOGGER = logging.getLogger(__name__)

# Maximum distinct values to enumerate for low-cardinality text columns
LOW_CARDINALITY_THRESHOLD = 20
SAMPLE_VALUES_COUNT = 5


def build_schema_profile(database_path: str, table_name: str) -> dict[str, Any]:
    """Generate a rich profile for a single DuckDB table.

    Returns
    -------
    dict with keys: table_name, row_count, columns (list of column profiles).
    Each column profile has: name, type, null_pct, distinct_count,
    min, max, sample_values.
    """
    with duckdb.connect(database_path, read_only=True) as conn:
        row_count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
        col_info = conn.execute(f'DESCRIBE "{table_name}"').fetchall()

        columns: list[dict[str, Any]] = []
        for col_name, col_type, *_ in col_info:
            profile = _profile_column(conn, table_name, col_name, col_type, row_count)
            columns.append(profile)

    return {
        "table_name": table_name,
        "row_count": row_count,
        "columns": columns,
    }


def profile_to_prompt_text(profile: dict[str, Any]) -> str:
    """Render a schema profile as compact text for injection into LLM prompts."""
    lines: list[str] = []
    lines.append(f"Table: {profile['table_name']} ({profile['row_count']} rows)")
    for col in profile["columns"]:
        parts = [f"  {col['name']} ({col['type']})"]
        parts.append(f"null={col['null_pct']:.0f}%")
        parts.append(f"distinct={col['distinct_count']}")
        if col.get("min") is not None:
            parts.append(f"range=[{col['min']}..{col['max']}]")
        if col.get("sample_values"):
            samples = ", ".join(str(v) for v in col["sample_values"][:SAMPLE_VALUES_COUNT])
            parts.append(f"examples=[{samples}]")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def all_tables_prompt_text(database_path: str, exclude_raw: bool = True) -> str:
    """Build prompt text covering all user tables in the database."""
    with duckdb.connect(database_path, read_only=True) as conn:
        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]

    if not tables:
        return "No tables are loaded."

    profiles: list[str] = []
    for table in tables:
        if exclude_raw and table.endswith("__raw"):
            continue
        try:
            profile = build_schema_profile(database_path, table)
            profiles.append(profile_to_prompt_text(profile))
        except Exception:
            LOGGER.warning("Failed to profile table %s", table, exc_info=True)
    return "\n\n".join(profiles) if profiles else "No tables are loaded."


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _profile_column(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    col_name: str,
    col_type: str,
    total_rows: int,
) -> dict[str, Any]:
    """Build the profile dict for a single column."""
    profile: dict[str, Any] = {
        "name": col_name,
        "type": col_type,
        "null_pct": 0.0,
        "distinct_count": 0,
        "min": None,
        "max": None,
        "sample_values": [],
    }
    if total_rows == 0:
        return profile

    # Null % and distinct count
    quoted = f'"{col_name}"'
    stats = conn.execute(
        f"SELECT "
        f"  ROUND(100.0 * SUM(CASE WHEN {quoted} IS NULL THEN 1 ELSE 0 END) / COUNT(*), 1) AS null_pct, "
        f"  COUNT(DISTINCT {quoted}) AS distinct_count "
        f'FROM "{table}"'
    ).fetchone()
    profile["null_pct"] = float(stats[0] or 0)
    profile["distinct_count"] = int(stats[1] or 0)

    # Min/max for numeric and date types
    upper_type = col_type.upper()
    is_numeric = any(t in upper_type for t in ("INT", "DOUBLE", "FLOAT", "DECIMAL", "NUMERIC", "BIGINT", "SMALLINT", "TINYINT", "HUGEINT"))
    is_date = any(t in upper_type for t in ("DATE", "TIMESTAMP", "TIME"))

    if is_numeric or is_date:
        try:
            minmax = conn.execute(
                f"SELECT MIN({quoted}), MAX({quoted}) FROM \"{table}\" WHERE {quoted} IS NOT NULL"
            ).fetchone()
            if minmax:
                profile["min"] = _safe_value(minmax[0])
                profile["max"] = _safe_value(minmax[1])
        except Exception:
            pass

    # Sample / enumeration for low-cardinality text columns
    if profile["distinct_count"] <= LOW_CARDINALITY_THRESHOLD and profile["distinct_count"] > 0:
        try:
            rows = conn.execute(
                f"SELECT DISTINCT {quoted} FROM \"{table}\" WHERE {quoted} IS NOT NULL ORDER BY {quoted} LIMIT {LOW_CARDINALITY_THRESHOLD}"
            ).fetchall()
            profile["sample_values"] = [_safe_value(r[0]) for r in rows]
        except Exception:
            pass
    elif profile["distinct_count"] > LOW_CARDINALITY_THRESHOLD:
        try:
            rows = conn.execute(
                f"SELECT DISTINCT {quoted} FROM \"{table}\" WHERE {quoted} IS NOT NULL LIMIT {SAMPLE_VALUES_COUNT}"
            ).fetchall()
            profile["sample_values"] = [_safe_value(r[0]) for r in rows]
        except Exception:
            pass

    return profile


def _safe_value(val: Any) -> Any:
    """Convert DuckDB return values to JSON-safe types."""
    if isinstance(val, (datetime, date)):
        return val.isoformat()
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    if hasattr(val, "item"):  # numpy scalar
        return val.item()
    return val

