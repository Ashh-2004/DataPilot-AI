"""Automatic dataset cleaning with databroom, safety rails, and pandas fallback.

Pipeline: databroom clean -> safety checks -> deduplication & empty drop -> type inference -> report.
If databroom fails, falls back to a minimal pandas cleaner.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safety thresholds
# ---------------------------------------------------------------------------
MAX_ROW_DROP_FRACTION = 0.20  # never silently drop >20% of rows
MIN_COLUMN_POPULATED_FRACTION = 0.50  # keep columns that are >50% populated


@dataclass
class CleaningResult:
    """Outcome of clean_dataset()."""

    cleaned_df: pd.DataFrame
    column_mapping: dict[str, str]  # raw name -> cleaned name
    report: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def clean_dataset(df: pd.DataFrame, source_name: str) -> CleaningResult:
    """Clean a raw DataFrame, apply safety rules, and infer types.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as read from the uploaded file.
    source_name : str
        Human-readable name of the source (used in log messages).

    Returns
    -------
    CleaningResult
        Contains the cleaned DataFrame, column mapping, report, and warnings.
    """
    warnings: list[str] = []
    original_shape = df.shape
    original_columns = list(df.columns)
    nulls_before = {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()}
    dupes_before = int(df.duplicated().sum())

    # --- Step 1: Attempt databroom cleaning --------------------------------
    databroom_success = False
    try:
        from databroom import Broom
        broom = Broom(df.copy())
        # Use clean_rows and standardize
        broom.clean_all()
        intermediate_df = broom.get_df()
        history = broom.get_history()
        for entry in history:
            shape_change = entry.get("shape_change", {})
            before = shape_change.get("before", ())
            after = shape_change.get("after", ())
            if before and after and before != after:
                warnings.append(f"{entry.get('function', 'databroom')}: shape {before} -> {after}")
        databroom_success = True
        LOGGER.info("databroom cleaning succeeded for %s", source_name)
    except Exception as exc:
        LOGGER.warning("databroom failed for %s (%s); falling back to pandas cleaner", source_name, exc)
        warnings.append(f"databroom cleaning failed ({exc}); used pandas fallback cleaner.")
        intermediate_df = df.copy()

    # --- Step 2: Safety check on row dropping ------------------------------
    rows_dropped_so_far = len(df) - len(intermediate_df)
    if len(df) > 0 and (rows_dropped_so_far / len(df)) > MAX_ROW_DROP_FRACTION:
        LOGGER.warning(
            "Cleaning %s would drop %d/%d rows (%.0f%%); skipping row drop step",
            source_name, rows_dropped_so_far, len(df), 100 * rows_dropped_so_far / len(df),
        )
        warnings.append(
            f"Row cleaning skipped: would remove {rows_dropped_so_far}/{len(df)} rows "
            f"({100 * rows_dropped_so_far / len(df):.0f}%), exceeding the 20% safety limit."
        )
        # Restore rows from original df
        intermediate_df = df.copy()

    # --- Step 3: Normalize and sanitize column names cleanly ---------------
    # Build clean snake_case column names from original df to maintain 1:1 mapping
    cleaned_col_names = []
    col_mapping: dict[str, str] = {}
    used_names: set[str] = set()

    for orig_col in original_columns:
        clean_name = _clean_column_identifier(str(orig_col))
        # Handle collision between sanitized column names
        base_name = clean_name
        counter = 2
        while clean_name in used_names:
            clean_name = f"{base_name}_{counter}"
            counter += 1
        used_names.add(clean_name)
        cleaned_col_names.append(clean_name)
        col_mapping[orig_col] = clean_name

    # Align columns from intermediate_df or original df
    cleaned_df = pd.DataFrame()
    for orig_col, clean_col in zip(original_columns, cleaned_col_names):
        # Check if databroom dropped this column
        is_dropped = False
        if databroom_success and orig_col not in intermediate_df.columns:
            # Check if an altered version exists in intermediate_df
            matched_col = None
            for cand in intermediate_df.columns:
                if _clean_column_identifier(cand) == clean_col:
                    matched_col = cand
                    break
            if matched_col is not None:
                cleaned_df[clean_col] = intermediate_df[matched_col]
            else:
                is_dropped = True
        else:
            cleaned_df[clean_col] = intermediate_df[orig_col] if orig_col in intermediate_df.columns else df[orig_col]

        # --- Step 4: Safety check on column dropping (>50% populated) ------
        if is_dropped:
            pop_rate = df[orig_col].notna().mean()
            if pop_rate >= MIN_COLUMN_POPULATED_FRACTION:
                # Safety guard: restore column
                cleaned_df[clean_col] = df[orig_col]
                warnings.append(
                    f"Column '{orig_col}' was restored (was {pop_rate:.0%} populated, "
                    f"above the 50% safety threshold)."
                )
                LOGGER.info("Restored column %s for %s (populated %.1f%%)", orig_col, source_name, pop_rate * 100)
            else:
                LOGGER.info("Dropped sparse column %s for %s (populated %.1f%%)", orig_col, source_name, pop_rate * 100)

    # --- Step 5: Post-clean row & whitespace hygiene -----------------------
    # Strip whitespace from text columns
    for col in cleaned_df.select_dtypes(include=["object", "string"]).columns:
        cleaned_df[col] = cleaned_df[col].map(lambda x: x.strip() if isinstance(x, str) else x)

    # Drop completely empty rows (all null or whitespace)
    cleaned_df = cleaned_df.dropna(how="all").reset_index(drop=True)

    # Drop exact duplicate rows
    dupes_pre_dedupe = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = dupes_pre_dedupe - len(cleaned_df)

    # Drop completely empty columns
    empty_cols_dropped = []
    for col in list(cleaned_df.columns):
        if cleaned_df[col].dropna().empty:
            cleaned_df.drop(columns=[col], inplace=True)
            empty_cols_dropped.append(col)

    # --- Step 6: Post-clean type inference ---------------------------------
    cleaned_df = _infer_types(cleaned_df)

    # --- Step 7: Metrics & Report ------------------------------------------
    nulls_after = {str(k): int(v) for k, v in cleaned_df.isnull().sum().to_dict().items()}
    raw_score, raw_issues = compute_quality_score_and_issues(df)
    clean_score, clean_issues = compute_quality_score_and_issues(cleaned_df)

    report: dict[str, Any] = {
        "source_name": source_name,
        "rows_before": original_shape[0],
        "rows_after": len(cleaned_df),
        "columns_before": original_shape[1],
        "columns_after": len(cleaned_df.columns),
        "duplicates_removed": max(duplicates_removed, dupes_before),
        "empty_columns_dropped": empty_cols_dropped,
        "nulls_per_column_before": nulls_before,
        "nulls_per_column_after": nulls_after,
        "columns_renamed": {k: v for k, v in col_mapping.items() if k != v},
        "operations": _describe_operations(databroom_success, warnings),
        "raw_score": raw_score,
        "clean_score": clean_score,
        "quality_score": clean_score,  # backward compatibility
        "issues": raw_issues if raw_issues else clean_issues,
        "clean_issues": clean_issues,
    }

    return CleaningResult(
        cleaned_df=cleaned_df,
        column_mapping=col_mapping,
        report=report,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Fallback cleaner
# ---------------------------------------------------------------------------

def _pandas_fallback_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal pandas cleaning when databroom is bypassed or fails."""
    result = df.copy()
    for col in result.select_dtypes(include=["object", "string"]).columns:
        result[col] = result[col].map(lambda x: x.strip() if isinstance(x, str) else x)
    result.columns = [_clean_column_identifier(str(c)) for c in result.columns]
    result = result.dropna(how="all").reset_index(drop=True)
    result = result.dropna(axis=1, how="all")
    result = result.drop_duplicates().reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------

_BOOL_MAP = {
    "true": True, "false": False, "yes": True, "no": False,
    "1": True, "0": False, "t": True, "f": False, "y": True, "n": False,
}

_CURRENCY_RE = re.compile(r"^[\$€£₹¥]?\s*[\d,]+\.?\d*$")


def _infer_types(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date-like columns, numeric strings with currency/commas, and booleans."""
    result = df.copy()
    for col in result.columns:
        if result[col].dtype == object or pd.api.types.is_string_dtype(result[col]):
            non_null = result[col].dropna()
            if non_null.empty:
                continue

            # Boolean detection
            lowered = non_null.astype(str).str.strip().str.lower()
            if len(lowered) > 0 and lowered.isin(_BOOL_MAP.keys()).all():
                result[col] = result[col].astype(str).str.strip().str.lower().map(_BOOL_MAP)
                continue

            # Date detection (try sample first)
            sample = non_null.head(20)
            try:
                parsed = pd.to_datetime(sample, format="mixed", errors="coerce")
                if parsed.notna().sum() >= len(sample) * 0.8:
                    result[col] = pd.to_datetime(result[col], format="mixed", errors="coerce")
                    continue
            except Exception:
                pass

            # Numeric strings with commas or currency symbols
            stripped = non_null.astype(str).str.strip()
            currency_match = stripped.apply(lambda x: bool(_CURRENCY_RE.match(x)))
            if currency_match.mean() > 0.8:
                cleaned_numeric = (
                    result[col].astype(str)
                    .str.replace(r"[\$€£₹¥,\s]", "", regex=True)
                )
                result[col] = pd.to_numeric(cleaned_numeric, errors="coerce")
                continue
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_column_identifier(name: str) -> str:
    """Normalize a column name into clean snake_case identifier."""
    s = str(name).strip()
    s = re.sub(r"[^\w\s]", "_", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_").lower()
    if not s or s[0].isdigit():
        s = f"col_{s}" if s else "col"
    return s


def sanitize_table_name(filename: str, existing_tables: list[str] | None = None) -> str:
    """Derive a clean, collision-free DuckDB table name from a filename.

    Rules:
    - Lowercase snake_case, no spaces or special characters.
    - Must not start with a digit.
    - Handles collisions by appending _2, _3, etc.
    """
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    stem = re.sub(r"[^\w\s]", "_", stem)
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_").lower()
    if not stem or stem[0].isdigit():
        stem = f"tbl_{stem}" if stem else "uploaded_table"

    name = stem
    if existing_tables:
        base = name
        counter = 2
        while name in existing_tables or f"{name}__raw" in existing_tables:
            name = f"{base}_{counter}"
            counter += 1

    return name


FINANCIAL_QTY_KEYWORDS = ("amount", "price", "qty", "quantity", "cost", "total", "sales", "revenue", "units", "balance")


def compute_quality_score_and_issues(df: pd.DataFrame) -> tuple[int, list[dict[str, Any]]]:
    """Compute documented data quality score (0-100) and list of detected issues.

    Scoring weights (no free points for column naming):
    - Completeness (non-null cell ratio): 25 pts
    - Duplicate rate: 20 pts
    - Invalid dates (strings that failed date parsing): 15 pts
    - Numeric columns stored as text: 15 pts
    - Empty and constant columns: 15 pts
    - Negative values in amount/price/qty columns: 10 pts
    """
    if df.empty:
        return 0, [{
            "severity": "high",
            "column": None,
            "description": "Dataset is completely empty.",
            "suggestion": "Upload a non-empty CSV/JSON/Excel/Parquet file.",
        }]

    issues: list[dict[str, Any]] = []
    total_rows = len(df)
    total_cols = len(df.columns)

    # 1. Completeness (25 pts)
    non_null_ratio = float(df.notna().mean().mean()) if total_cols > 0 else 0.0
    completeness_score = non_null_ratio * 25.0

    for col in df.columns:
        col_null_pct = float(df[col].isnull().mean() * 100)
        if col_null_pct > 10.0 and col_null_pct < 100.0:
            severity = "high" if col_null_pct >= 50.0 else "medium"
            issues.append({
                "severity": severity,
                "column": str(col),
                "description": f"Column '{col}' has {col_null_pct:.1f}% missing values.",
                "suggestion": "Impute missing values or remove sparse rows.",
            })

    # 2. Duplicate Rate (20 pts)
    dupe_count = int(df.duplicated().sum())
    dupe_rate = (dupe_count / total_rows) if total_rows > 0 else 0.0
    duplicate_score = (1.0 - dupe_rate) * 20.0

    if dupe_count > 0:
        issues.append({
            "severity": "medium",
            "column": None,
            "description": f"Found {dupe_count} duplicate row(s) ({dupe_rate:.1%} of dataset).",
            "suggestion": "Deduplicate rows to prevent skewed aggregations.",
        })

    # 3. Invalid Dates (15 pts)
    invalid_date_count = 0
    total_date_candidates = 0
    date_cols_evaluated: set[str] = set()

    for col in df.columns:
        col_lower = str(col).lower()
        s = df[col].dropna()
        if s.empty:
            continue

        is_date_col = any(k in col_lower for k in ("date", "time", "created", "updated", "joined", "timestamp"))
        if not is_date_col and (s.dtype == object or pd.api.types.is_string_dtype(s)):
            sample = s.head(20).astype(str)
            try:
                parsed_sample = pd.to_datetime(sample, errors="coerce", format="mixed")
                if parsed_sample.notna().sum() >= max(1, len(sample) * 0.4):
                    is_date_col = True
            except Exception:
                pass

        if is_date_col and (s.dtype == object or pd.api.types.is_string_dtype(s)):
            date_cols_evaluated.add(str(col))
            str_vals = s.astype(str)
            parsed = pd.to_datetime(str_vals, errors="coerce", format="mixed")
            failed = int(parsed.isna().sum())
            invalid_date_count += failed
            total_date_candidates += len(str_vals)

            if failed > 0:
                issues.append({
                    "severity": "high",
                    "column": str(col),
                    "description": f"Column '{col}' contains {failed} date value(s) that failed date parsing.",
                    "suggestion": "Standardize date strings into ISO format (YYYY-MM-DD).",
                })

    invalid_date_frac = (invalid_date_count / total_date_candidates) if total_date_candidates > 0 else 0.0
    invalid_date_score = (1.0 - invalid_date_frac) * 15.0

    # 4. Numeric Columns Stored as Text (15 pts)
    text_numeric_cols = 0
    for col in df.columns:
        if str(col) in date_cols_evaluated:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            str_vals = s.astype(str).str.strip().str.replace(r"[\$€£₹¥,\s]", "", regex=True)
            num_parsed = pd.to_numeric(str_vals, errors="coerce")
            if num_parsed.notna().sum() >= max(1, len(s) * 0.8):
                text_numeric_cols += 1
                issues.append({
                    "severity": "medium",
                    "column": str(col),
                    "description": f"Column '{col}' contains numeric values stored as text/string.",
                    "suggestion": "Cast column to numeric (int/float) data type.",
                })

    text_num_frac = (text_numeric_cols / total_cols) if total_cols > 0 else 0.0
    numeric_text_score = (1.0 - text_num_frac) * 15.0

    # 5. Empty and Constant Columns (15 pts)
    empty_cols = 0
    constant_cols = 0
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            empty_cols += 1
            issues.append({
                "severity": "high",
                "column": str(col),
                "description": f"Column '{col}' is completely empty (100% missing values).",
                "suggestion": "Drop empty column from dataset.",
            })
        elif s.nunique() <= 1:
            constant_cols += 1
            issues.append({
                "severity": "low",
                "column": str(col),
                "description": f"Column '{col}' contains a single constant value.",
                "suggestion": "Check if constant column can be dropped.",
            })

    empty_const_frac = ((empty_cols + constant_cols) / total_cols) if total_cols > 0 else 0.0
    empty_constant_score = (1.0 - empty_const_frac) * 15.0

    # 6. Negative Values in Financial/Quantity Columns (10 pts)
    neg_cols = 0
    for col in df.columns:
        col_lower = str(col).lower()
        if any(k in col_lower for k in FINANCIAL_QTY_KEYWORDS):
            s = df[col].dropna()
            if not s.empty:
                if pd.api.types.is_numeric_dtype(s):
                    num_s = s
                else:
                    str_vals = s.astype(str).str.strip().str.replace(r"[\$€£₹¥,\s]", "", regex=True)
                    num_s = pd.to_numeric(str_vals, errors="coerce").dropna()

                neg_count = int((num_s < 0).sum())
                if neg_count > 0:
                    neg_cols += 1
                    issues.append({
                        "severity": "high",
                        "column": str(col),
                        "description": f"Column '{col}' contains {neg_count} negative value(s).",
                        "suggestion": "Verify whether negative values represent invalid records or returns.",
                    })

    neg_frac = (neg_cols / total_cols) if total_cols > 0 else 0.0
    negative_score = (1.0 - neg_frac) * 10.0

    final_score = int(round(
        completeness_score + duplicate_score + invalid_date_score +
        numeric_text_score + empty_constant_score + negative_score
    ))
    final_score = min(100, max(0, final_score))

    return final_score, issues


def _quality_score(df: pd.DataFrame) -> int:
    """Legacy wrapper for quality score."""
    score, _ = compute_quality_score_and_issues(df)
    return score


def _describe_operations(databroom_used: bool, warnings: list[str]) -> list[str]:
    """List operations applied."""
    prefix = "databroom_" if databroom_used else "pandas_"
    ops = [
        f"{prefix}clean",
        "clean_column_names",
        "drop_empty_rows",
        "drop_duplicates",
        "type_inference",
    ]
    return ops


class DataCleaner:
    """Class interface wrapper for dataset cleaning."""

    @staticmethod
    def clean(df: pd.DataFrame, source_name: str = "upload") -> pd.DataFrame:
        """Clean raw DataFrame and return cleaned DataFrame."""
        result = clean_dataset(df, source_name)
        return result.cleaned_df

