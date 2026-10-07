"""Unit tests for SQL guardrails using sqlglot AST verification."""

import pytest
from app.services.sql_guardrails import (
    validate_sql,
    SQLValidationError,
    DEFAULT_ROW_LIMIT,
    MAX_ROW_LIMIT,
)


def test_valid_select_queries():
    """Basic SELECT and WITH queries pass validation and have LIMIT added."""
    sql1 = "SELECT * FROM sales_data"
    val1 = validate_sql(sql1)
    assert "LIMIT" in val1
    assert "sales_data" in val1

    sql2 = "WITH cte AS (SELECT * FROM sales_data) SELECT * FROM cte"
    val2 = validate_sql(sql2)
    assert "LIMIT" in val2


def test_enforces_existing_limit():
    """Existing reasonable limits are preserved; excessive limits are capped."""
    sql1 = "SELECT * FROM sales_data LIMIT 10"
    val1 = validate_sql(sql1)
    assert "10" in val1

    # Over 50k limit capped
    sql2 = "SELECT * FROM sales_data LIMIT 100000"
    val2 = validate_sql(sql2)
    assert str(MAX_ROW_LIMIT) in val2


def test_forbids_ddl_and_dml():
    """Blocks INSERT, UPDATE, DELETE, DROP, CREATE, ALTER."""
    dangerous = [
        "DROP TABLE sales_data",
        "DELETE FROM sales_data WHERE order_id = 1",
        "INSERT INTO sales_data VALUES (1, '2025-01-01')",
        "UPDATE sales_data SET total_amount = 0",
        "CREATE TABLE hacked AS SELECT 1",
        "ALTER TABLE sales_data DROP COLUMN order_id",
    ]
    for sql in dangerous:
        with pytest.raises(SQLValidationError):
            validate_sql(sql)


def test_forbids_dangerous_commands():
    """Blocks COPY, ATTACH, INSTALL, PRAGMA."""
    commands = [
        "COPY sales_data TO 'leak.csv'",
        "ATTACH 'other.db' AS other",
        "INSTALL httpfs",
        "PRAGMA database_list",
        "EXPORT DATABASE 'backup'",
    ]
    for sql in commands:
        with pytest.raises(SQLValidationError):
            validate_sql(sql)


def test_forbids_multiple_statements():
    """Rejects stacked statements separated by semicolons."""
    multi = "SELECT * FROM sales_data; DROP TABLE sales_data;"
    with pytest.raises(SQLValidationError):
        validate_sql(multi)

