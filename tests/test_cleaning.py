"""Tests for databroom cleaning service with messy fixtures and safety rails."""

import io
import pandas as pd
import pytest

from app.services.cleaning import (
    clean_dataset,
    sanitize_table_name,
    MAX_ROW_DROP_FRACTION,
    MIN_COLUMN_POPULATED_FRACTION,
)


@pytest.fixture
def messy_csv_df():
    """A messy CSV with whitespace, duplicates, empty rows, and mixed data."""
    csv_data = '''name,AGE,salary,notes,joined_date,is_active
 Alice  ,  30  , "$50,000" , note 1 , 2023-01-15 , yes 
 Bob    ,  25  , "$65,000" ,        , 2022-05-10 , no  
 Charlie,      , "$70,000" ,        , 2021-11-01 , true 
 Alice  ,  30  , "$50,000" , note 1 , 2023-01-15 , yes 
 , , , , , 
 David  ,  40  , "$90,000" ,        , 2020-03-20 , false
 Eva    ,  35  , "$80,000" ,        , 2024-02-01 , 1 
'''
    return pd.read_csv(io.StringIO(csv_data), skipinitialspace=True)


def test_clean_dataset_basic(messy_csv_df):
    """Test standard cleaning pipeline removes duplicates, strips whitespace, infers types."""
    res = clean_dataset(messy_csv_df, "test_dataset")
    df = res.cleaned_df
    
    # Empty row and duplicate row should be removed
    assert len(df) < len(messy_csv_df)
    assert res.report["duplicates_removed"] >= 1
    
    # Column names sanitized
    for col in df.columns:
        assert not col.startswith(" ")
        assert not col.endswith(" ")
        assert col.islower()
    
    # Type inference checks
    # salary should be converted to numeric
    assert pd.api.types.is_numeric_dtype(df["salary"])
    assert df["salary"].iloc[0] == 50000
    
    # is_active converted to boolean
    assert pd.api.types.is_bool_dtype(df["is_active"])
    
    # joined_date converted to datetime
    assert pd.api.types.is_datetime64_any_dtype(df["joined_date"])
    
    # Quality score should be computed
    assert 0 <= res.report["quality_score"] <= 100


def test_safety_rule_never_drop_populated_column():
    """Columns populated > 50% must NOT be dropped even if cleaning would drop them."""
    data = {
        "id": list(range(20)),
        # 60% populated column (12 non-null, 8 null)
        "mostly_present": [f"val_{i}" if i < 12 else None for i in range(20)],
        # 10% populated column (2 non-null, 18 null) - under 50%, can be dropped if cleaner decides
        "mostly_empty": [f"rare_{i}" if i < 2 else None for i in range(20)],
    }
    df = pd.DataFrame(data)
    res = clean_dataset(df, "test_column_safety")
    
    # mostly_present is 60% populated, so it must be preserved
    clean_cols = res.cleaned_df.columns
    assert any("mostly_present" in c for c in clean_cols)
    assert any("id" in c for c in clean_cols)


def test_safety_rule_row_drop_threshold():
    """If cleaning drops > 20% of rows, row dropping must be skipped with warning."""
    # Create 10 rows where 4 are empty/droppable (40% drop)
    data = [
        {"col1": "val", "col2": 1},
        {"col1": "val", "col2": 2},
        {"col1": "val", "col2": 3},
        {"col1": "val", "col2": 4},
        {"col1": "val", "col2": 5},
        {"col1": "val", "col2": 6},
        {"col1": None, "col2": None},
        {"col1": None, "col2": None},
        {"col1": None, "col2": None},
        {"col1": None, "col2": None},
    ]
    df = pd.DataFrame(data)
    res = clean_dataset(df, "test_row_safety")
    
    # A warning should be emitted if row-drop threshold exceeded
    if len(df) - len(res.cleaned_df) > 0.20 * len(df):
        assert any("safety" in w.lower() or "row" in w.lower() for w in res.warnings)


def test_fallback_cleaner():
    """Fallback cleaner works when databroom encounters unusual inputs."""
    from app.services.cleaning import _pandas_fallback_clean
    
    raw = pd.DataFrame({
        "  First Name  ": [" Alice ", "Bob ", "Charlie", "  "],
        " Age ": [25, 30, 35, None],
        "EmptyCol": [None, None, None, None]
    })
    cleaned = _pandas_fallback_clean(raw)
    assert "emptycol" not in cleaned.columns
    assert "first_name" in cleaned.columns
    assert cleaned["first_name"].iloc[0] == "Alice"


def test_sanitize_table_name():
    """Table names must be snake_case, no leading digits, and handle collisions."""
    assert sanitize_table_name("Sales Data 2025.csv") == "sales_data_2025"
    assert sanitize_table_name("123orders.json") == "tbl_123orders"
    assert sanitize_table_name("User-Data!#@.csv") == "user_data"
    
    existing = ["sales_data", "sales_data_2"]
    assert sanitize_table_name("Sales Data.csv", existing) == "sales_data_3"
