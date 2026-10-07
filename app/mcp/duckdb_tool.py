"""DuckDB MCP tools for uploads, schema inspection, and read-only queries."""

import json
import logging
import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from app.services.cleaning import CleaningResult, clean_dataset, sanitize_table_name
from app.services.schema_profile import all_tables_prompt_text, build_schema_profile
from app.services.sql_guardrails import (
    QUERY_TIMEOUT_SECONDS,
    SQLValidationError,
    explain_sql,
    validate_sql,
)

LOGGER = logging.getLogger(__name__)
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DuckDBTool:
    """Owns all access to the embedded DuckDB database."""

    def __init__(self, database_path: str) -> None:
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        self.database_path = database_path
        self.last_cleaning_report: dict[str, Any] = {}
        self.last_dataset_report: dict[str, Any] = {}
        self._ensure_metadata_tables()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(self.database_path)

    def _ensure_metadata_tables(self) -> None:
        """Create metadata tables for cleaning reports, column mappings, and dataset reports if missing."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _cleaning_reports (
                    table_name VARCHAR PRIMARY KEY,
                    report JSON,
                    created_at TIMESTAMP DEFAULT current_timestamp
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _column_mappings (
                    table_name VARCHAR PRIMARY KEY,
                    mapping JSON,
                    created_at TIMESTAMP DEFAULT current_timestamp
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _dataset_reports (
                    table_name VARCHAR,
                    version INTEGER,
                    report JSON,
                    created_at TIMESTAMP DEFAULT current_timestamp,
                    PRIMARY KEY (table_name, version)
                )
            """)

    # ------------------------------------------------------------------
    # Table listing
    # ------------------------------------------------------------------

    def list_tables(self) -> list[str]:
        """Return user tables in the database (excluding metadata tables)."""
        with self._connect() as conn:
            all_tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
        return [t for t in all_tables if not t.startswith("_")]

    def list_user_tables(self) -> list[str]:
        """Return user tables excluding __raw backup tables, ordered with most recent first."""
        with self._connect() as conn:
            try:
                rows = conn.execute(
                    "SELECT table_name FROM _cleaning_reports ORDER BY created_at DESC"
                ).fetchall()
                recent_tables = [r[0] for r in rows if not r[0].startswith("_") and not r[0].endswith("__raw")]
                all_tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
                user_tables = [t for t in all_tables if not t.startswith("_") and not t.endswith("__raw")]
                ordered = [t for t in recent_tables if t in user_tables]
                for t in user_tables:
                    if t not in ordered:
                        ordered.append(t)
                return ordered
            except Exception:
                all_tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
                return [t for t in all_tables if not t.startswith("_") and not t.endswith("__raw")]

    def reset_database(self) -> None:
        """Drop all user tables and clear metadata to reset DuckDB completely."""
        with self._connect() as conn:
            all_tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall() if not row[0].startswith("_")]
            for table in all_tables:
                conn.execute(f'DROP TABLE IF EXISTS "{table}"')
            try:
                conn.execute("DELETE FROM _cleaning_reports")
                conn.execute("DELETE FROM _column_mappings")
                conn.execute("DELETE FROM _dataset_reports")
            except Exception:
                pass
        self.last_cleaning_report = {}
        self.last_dataset_report = {}

    # ------------------------------------------------------------------
    # Read-only query execution
    # ------------------------------------------------------------------

    def execute_read_only(self, sql: str) -> list[dict[str, Any]]:
        """Execute one validated read-only query and return JSON-compatible records."""
        validated_sql = validate_sql(sql)
        with duckdb.connect(self.database_path, read_only=True) as connection:
            frame = connection.execute(validated_sql).fetchdf()
        return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")

    def explain_query(self, sql: str) -> str | None:
        """Run EXPLAIN to pre-validate SQL. Returns error message or None."""
        try:
            clean_sql = validate_sql(sql)
        except SQLValidationError as exc:
            return str(exc)
        with duckdb.connect(self.database_path, read_only=True) as connection:
            return explain_sql(clean_sql, connection)

    # ------------------------------------------------------------------
    # File loading with cleaning pipeline & dataset report assembly
    # ------------------------------------------------------------------

    def load_file(
        self,
        file_path: str,
        table_name: str,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
    ) -> int:
        """Load a file into DuckDB with the full cleaning pipeline and report assembly.

        Pipeline: read -> keep raw -> databroom clean -> type inference -> load -> report.
        Raw data is preserved in <table_name>__raw.
        """
        if not _IDENTIFIER.fullmatch(table_name):
            raise ValueError(f"Invalid table name: {table_name!r}")

        suffix = Path(file_path).suffix.lower()
        supported = {".csv", ".json", ".xlsx", ".xls", ".parquet"}
        if suffix not in supported:
            raise ValueError(f"Unsupported file type: {suffix}")

        # --- Read the raw file ------------------------------------------------
        original_frame = self._read_file(file_path, suffix)

        # --- Store the raw table first ----------------------------------------
        raw_table = f"{table_name}__raw"
        with self._connect() as conn:
            conn.register("_raw_upload", original_frame)
            conn.execute(f'CREATE OR REPLACE TABLE "{raw_table}" AS SELECT * FROM _raw_upload')

        # --- Clean with safety rails ------------------------------------------
        result: CleaningResult = clean_dataset(original_frame, table_name)
        self.last_cleaning_report = result.report

        # --- Load cleaned data ------------------------------------------------
        with self._connect() as conn:
            conn.register("_cleaned_upload", result.cleaned_df)
            conn.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM _cleaned_upload')

        # --- Persist metadata & assemble dataset report -----------------------
        self._save_metadata(table_name, result)
        self.generate_and_save_dataset_report(table_name, result, model=model, base_url=base_url)

        LOGGER.info(
            "Loaded table %s: %d rows (%d raw), quality=%d/100",
            table_name, len(result.cleaned_df), len(original_frame),
            result.report.get("quality_score", 0),
        )
        return len(result.cleaned_df)

    def generate_and_save_dataset_report(
        self,
        table_name: str,
        result: CleaningResult,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
    ) -> dict[str, Any]:
        """Run profiling + quality + domain classification, assemble report, and persist to _dataset_reports table."""
        from app.services.domain_classifier import classify_domain

        profile = build_schema_profile(self.database_path, table_name)

        with duckdb.connect(self.database_path, read_only=True) as conn:
            sample_df = conn.execute(f'SELECT * FROM "{table_name}" LIMIT 5').fetchdf()
            sample_rows = sample_df.astype(object).where(pd.notna(sample_df), None).to_dict(orient="records")

        columns = [c["name"] for c in profile.get("columns", [])]
        dtypes = {c["name"]: c["type"] for c in profile.get("columns", [])}
        domain_info = classify_domain(columns, dtypes, sample_rows, model=model, base_url=base_url)

        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM _dataset_reports WHERE table_name = ?",
                [table_name],
            ).fetchone()
            version = (row[0] if row else 0) + 1

        report: dict[str, Any] = {
            "table_name": table_name,
            "version": version,
            "rows_raw": result.report.get("rows_before", profile.get("row_count", 0)),
            "rows_clean": profile.get("row_count", 0),
            "columns_count": len(columns),
            "quality": {
                "raw_score": result.report.get("raw_score", 0),
                "clean_score": result.report.get("clean_score", 0),
                "issues": result.report.get("issues", []),
            },
            "domain": domain_info,
            "profile": profile,
            "cleaning_summary": result.report,
        }

        with self._connect() as conn:
            conn.execute(
                """INSERT INTO _dataset_reports (table_name, version, report, created_at)
                   VALUES (?, ?, ?::JSON, current_timestamp)""",
                [table_name, version, json.dumps(report, default=str)],
            )

        self.last_dataset_report = report
        return report

    def get_dataset_report(self, table_name: str, version: int | None = None) -> dict[str, Any] | None:
        """Retrieve dataset report for a table, optionally for a specific version."""
        try:
            with self._connect() as conn:
                if version is not None:
                    row = conn.execute(
                        "SELECT report FROM _dataset_reports WHERE table_name = ? AND version = ?",
                        [table_name, version],
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT report FROM _dataset_reports WHERE table_name = ? ORDER BY version DESC LIMIT 1",
                        [table_name],
                    ).fetchone()
                if row and row[0]:
                    return json.loads(row[0])
        except Exception:
            LOGGER.warning("Failed to fetch dataset report for %s", table_name, exc_info=True)
        return None

    def get_all_dataset_summaries(self) -> list[dict[str, Any]]:
        """Return list of all loaded datasets with latest version and report summary."""
        user_tables = self.list_user_tables()
        summaries: list[dict[str, Any]] = []
        for tbl in user_tables:
            report = self.get_dataset_report(tbl)
            if report:
                summaries.append({
                    "table_name": tbl,
                    "version": report.get("version", 1),
                    "rows_clean": report.get("rows_clean", 0),
                    "domain": report.get("domain", {}).get("domain", "Unknown"),
                    "clean_score": report.get("quality", {}).get("clean_score", 0),
                    "report": report,
                })
            else:
                summaries.append({
                    "table_name": tbl,
                    "version": 1,
                    "rows_clean": 0,
                    "domain": "Unknown",
                    "clean_score": 0,
                    "report": None,
                })
        return summaries

    @staticmethod
    def _read_file(file_path: str, suffix: str) -> pd.DataFrame:
        """Read a file into a DataFrame based on its suffix."""
        if suffix == ".csv":
            return pd.read_csv(file_path)
        elif suffix == ".json":
            return pd.read_json(file_path)
        elif suffix in (".xlsx", ".xls"):
            return pd.read_excel(file_path)
        elif suffix == ".parquet":
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _save_metadata(self, table_name: str, result: CleaningResult) -> None:
        """Persist cleaning report and column mapping to DuckDB metadata tables."""
        try:
            report_json = json.dumps(result.report, default=str)
            mapping_json = json.dumps(result.column_mapping, default=str)
            with self._connect() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO _cleaning_reports (table_name, report, created_at)
                       VALUES (?, ?::JSON, current_timestamp)""",
                    [table_name, report_json],
                )
                conn.execute(
                    """INSERT OR REPLACE INTO _column_mappings (table_name, mapping, created_at)
                       VALUES (?, ?::JSON, current_timestamp)""",
                    [table_name, mapping_json],
                )
        except Exception:
            LOGGER.warning("Failed to persist metadata for %s", table_name, exc_info=True)

    # ------------------------------------------------------------------
    # Schema context for LLM prompts
    # ------------------------------------------------------------------

    def schema_context(self) -> str:
        """Return a rich schema profile for all user tables."""
        return all_tables_prompt_text(self.database_path, exclude_raw=True)

    def get_column_mapping(self, table_name: str) -> dict[str, str]:
        """Retrieve the raw->clean column mapping for a table."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT mapping FROM _column_mappings WHERE table_name = ?",
                    [table_name],
                ).fetchone()
                if row:
                    return json.loads(row[0])
        except Exception:
            pass
        return {}

    def get_all_column_names(self) -> list[str]:
        """Return all column names across all user tables (for fuzzy matching)."""
        columns: list[str] = []
        with self._connect() as conn:
            for table in self.list_user_tables():
                try:
                    cols = conn.execute(f'DESCRIBE "{table}"').fetchall()
                    columns.extend(row[0] for row in cols)
                except Exception:
                    pass
        return columns

    def dataset_overview(self) -> dict[str, Any]:
        """Return bounded metadata and a sample for conversational dataset walkthroughs."""
        tables = self.list_user_tables()
        overview: dict[str, Any] = {"tables": []}
        with duckdb.connect(self.database_path, read_only=True) as connection:
            for table in tables:
                try:
                    profile = build_schema_profile(self.database_path, table)
                    sample = connection.execute(f'SELECT * FROM "{table}" LIMIT 10').fetchdf()
                    overview["tables"].append(
                        {
                            "name": table,
                            "row_count": profile["row_count"],
                            "columns": profile["columns"],
                            "sample": sample.astype(object).where(pd.notna(sample), None).to_dict(orient="records"),
                        }
                    )
                except Exception:
                    LOGGER.warning("Failed to build overview for %s", table, exc_info=True)
        return overview
