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
    dict with keys: table_name, row_count, columns (list of column profiles), correlations.
    Each column profile has: name, type, null_pct, distinct_count, min, max,
    mean, median, std, p25, p75, sample_values, top_5, is_candidate_id, is_constant.
    """
    with duckdb.connect(database_path, read_only=True) as conn:
        row_count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
        col_info = conn.execute(f'DESCRIBE "{table_name}"').fetchall()

        columns: list[dict[str, Any]] = []
        for col_name, col_type, *_ in col_info:
            profile = _profile_column(conn, table_name, col_name, col_type, row_count)
            columns.append(profile)

        correlations = _compute_numeric_correlations(conn, table_name, columns)

    return {
        "table_name": table_name,
        "row_count": row_count,
        "columns": columns,
        "correlations": correlations,
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
        if col.get("mean") is not None:
            parts.append(f"mean={col['mean']}")
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
        "mean": None,
        "median": None,
        "std": None,
        "p25": None,
        "p75": None,
        "sample_values": [],
        "top_5": [],
        "is_candidate_id": False,
        "is_constant": False,
    }
    if total_rows == 0:
        return profile

    quoted = f'"{col_name}"'
    stats = conn.execute(
        f"SELECT "
        f"  ROUND(100.0 * SUM(CASE WHEN {quoted} IS NULL THEN 1 ELSE 0 END) / COUNT(*), 1) AS null_pct, "
        f"  COUNT(DISTINCT {quoted}) AS distinct_count "
        f'FROM "{table}"'
    ).fetchone()
    profile["null_pct"] = float(stats[0] or 0)
    profile["distinct_count"] = int(stats[1] or 0)

    # Candidate ID and Constant flags
    profile["is_constant"] = profile["distinct_count"] <= 1
    if total_rows > 0:
        profile["is_candidate_id"] = (profile["distinct_count"] / total_rows) >= 0.95

    upper_type = col_type.upper()
    is_numeric = any(t in upper_type for t in ("INT", "DOUBLE", "FLOAT", "DECIMAL", "NUMERIC", "BIGINT", "SMALLINT", "TINYINT", "HUGEINT"))
    is_date = any(t in upper_type for t in ("DATE", "TIMESTAMP", "TIME"))

    if is_numeric:
        try:
            row = conn.execute(
                f'SELECT '
                f'  MIN({quoted}), MAX({quoted}), '
                f'  AVG({quoted}), MEDIAN({quoted}), '
                f'  COALESCE(STDDEV_SAMP({quoted}), 0), '
                f'  QUANTILE_CONT({quoted}, 0.25), '
                f'  QUANTILE_CONT({quoted}, 0.75) '
                f'FROM "{table}" WHERE {quoted} IS NOT NULL'
            ).fetchone()
            if row and row[0] is not None:
                profile["min"] = _safe_value(row[0])
                profile["max"] = _safe_value(row[1])
                profile["mean"] = round(float(row[2]), 4) if row[2] is not None else None
                profile["median"] = _safe_value(row[3])
                profile["std"] = round(float(row[4]), 4) if row[4] is not None else None
                profile["p25"] = _safe_value(row[5])
                profile["p75"] = _safe_value(row[6])
        except Exception:
            pass
    elif is_date:
        try:
            minmax = conn.execute(
                f'SELECT MIN({quoted}), MAX({quoted}) FROM "{table}" WHERE {quoted} IS NOT NULL'
            ).fetchone()
            if minmax:
                profile["min"] = _safe_value(minmax[0])
                profile["max"] = _safe_value(minmax[1])
        except Exception:
            pass

    # Top-5 values with counts for categorical / non-numeric columns
    if not is_numeric:
        try:
            top_rows = conn.execute(
                f'SELECT {quoted}, COUNT(*) FROM "{table}" WHERE {quoted} IS NOT NULL GROUP BY {quoted} ORDER BY COUNT(*) DESC, {quoted} ASC LIMIT 5'
            ).fetchall()
            profile["top_5"] = [{"value": _safe_value(r[0]), "count": int(r[1])} for r in top_rows]
        except Exception:
            pass

    # Sample values
    if profile["distinct_count"] <= LOW_CARDINALITY_THRESHOLD and profile["distinct_count"] > 0:
        try:
            rows = conn.execute(
                f'SELECT DISTINCT {quoted} FROM "{table}" WHERE {quoted} IS NOT NULL ORDER BY {quoted} LIMIT {LOW_CARDINALITY_THRESHOLD}'
            ).fetchall()
            profile["sample_values"] = [_safe_value(r[0]) for r in rows]
        except Exception:
            pass
    elif profile["distinct_count"] > LOW_CARDINALITY_THRESHOLD:
        try:
            rows = conn.execute(
                f'SELECT DISTINCT {quoted} FROM "{table}" WHERE {quoted} IS NOT NULL LIMIT {SAMPLE_VALUES_COUNT}'
            ).fetchall()
            profile["sample_values"] = [_safe_value(r[0]) for r in rows]
        except Exception:
            pass

    return profile


def _compute_numeric_correlations(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    columns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compute top-10 numeric correlations ordered by absolute correlation value."""
    numeric_cols = [
        c["name"] for c in columns
        if any(t in c["type"].upper() for t in ("INT", "DOUBLE", "FLOAT", "DECIMAL", "NUMERIC", "BIGINT", "SMALLINT", "HUGEINT"))
    ]
    if len(numeric_cols) < 2:
        return []

    corrs: list[dict[str, Any]] = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            col1, col2 = numeric_cols[i], numeric_cols[j]
            try:
                val = conn.execute(
                    f'SELECT CORR("{col1}", "{col2}") FROM "{table_name}"'
                ).fetchone()[0]
                if val is not None and not (isinstance(val, float) and pd.isna(val)):
                    c_val = float(val)
                    corrs.append({
                        "col1": col1,
                        "col2": col2,
                        "correlation": round(c_val, 4),
                        "abs_correlation": round(abs(c_val), 4),
                    })
            except Exception:
                pass

    corrs.sort(key=lambda x: x["abs_correlation"], reverse=True)
    for item in corrs:
        item.pop("abs_correlation", None)
    return corrs[:10]


def _safe_value(val: Any) -> Any:
    """Convert DuckDB return values to JSON-safe types."""
    if isinstance(val, (datetime, date)):
        return val.isoformat()
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    if hasattr(val, "item"):  # numpy scalar
        return val.item()
    return val

