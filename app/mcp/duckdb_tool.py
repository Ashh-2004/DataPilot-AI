"""DuckDB MCP tools for uploads, schema inspection, and read-only queries."""

from pathlib import Path
import re
from typing import Any

import duckdb
import pandas as pd

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DuckDBTool:
    """Owns all access to the embedded DuckDB database."""

    def __init__(self, database_path: str) -> None:
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        self.database_path = database_path

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(self.database_path)

    def list_tables(self) -> list[str]:
        """Return user tables in the database."""
        with self._connect() as connection:
            return [row[0] for row in connection.execute("SHOW TABLES").fetchall()]

    def execute_read_only(self, sql: str) -> list[dict[str, Any]]:
        """Execute one read-only query and return JSON-compatible records."""
        normalized_sql = self._normalize_sql(sql)
        if not re.match(r"(?is)^(SELECT|WITH|DESCRIBE|SHOW)\b", normalized_sql):
            raise ValueError("Only SELECT, WITH, DESCRIBE, and SHOW queries are allowed.")
        with duckdb.connect(self.database_path, read_only=True) as connection:
            frame = connection.execute(normalized_sql).fetchdf()
        return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")

    @staticmethod
    def _normalize_sql(sql: str) -> str:
        """Remove model formatting and accept one optional trailing semicolon."""
        normalized_sql = re.sub(r"(?is)```(?:sql)?", "", sql).replace("```", "").strip()
        if not normalized_sql:
            raise ValueError("Planner returned empty SQL.")

        quote: str | None = None
        index = 0
        while index < len(normalized_sql):
            character = normalized_sql[index]
            if quote:
                if character == quote:
                    if index + 1 < len(normalized_sql) and normalized_sql[index + 1] == quote:
                        index += 1
                    else:
                        quote = None
            elif character in {"'", '"'}:
                quote = character
            elif character == ";":
                if normalized_sql[index + 1 :].strip():
                    raise ValueError("Only one SQL statement is allowed.")
                normalized_sql = normalized_sql[:index].rstrip()
                break
            index += 1
        return normalized_sql

    def load_file(self, file_path: str, table_name: str) -> int:
        """Load a CSV or JSON file into a sanitized DuckDB table."""
        if not _IDENTIFIER.fullmatch(table_name):
            raise ValueError("Invalid table name.")
        suffix = Path(file_path).suffix.lower()
        if suffix not in {".csv", ".json"}:
            raise ValueError("Only CSV and JSON uploads are supported.")
        reader = pd.read_csv if suffix == ".csv" else pd.read_json
        frame = reader(file_path)
        with self._connect() as connection:
            connection.register("uploaded_frame", frame)
            connection.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM uploaded_frame')
        return len(frame)

    def schema_context(self) -> str:
        """Return a compact schema description for the planner."""
        tables = self.list_tables()
        if not tables:
            return "No tables are loaded."
        descriptions: list[str] = []
        with self._connect() as connection:
            for table in tables:
                columns = connection.execute(f'DESCRIBE "{table}"').fetchall()
                descriptions.append(f"{table}: " + ", ".join(f"{row[0]} ({row[1]})" for row in columns))
        return "\n".join(descriptions)

    def dataset_overview(self) -> dict[str, Any]:
        """Return bounded metadata and a sample for conversational dataset walkthroughs."""
        tables = self.list_tables()
        overview: dict[str, Any] = {"tables": []}
        with duckdb.connect(self.database_path, read_only=True) as connection:
            for table in tables:
                columns = connection.execute(f'DESCRIBE "{table}"').fetchall()
                sample = connection.execute(f'SELECT * FROM "{table}" LIMIT 10').fetchdf()
                row_count = connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                overview["tables"].append(
                    {
                        "name": table,
                        "row_count": row_count,
                        "columns": [{"name": row[0], "type": row[1]} for row in columns],
                        "sample": sample.astype(object).where(pd.notna(sample), None).to_dict(orient="records"),
                    }
                )
        return overview
