"""Dataset profiler for universal column classification, summary statistics, and data quality scoring."""

import logging
import re
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class DatasetProfiler:
    """Profiles a pandas DataFrame to infer column semantics, statistical summaries, and data quality metrics."""

    @staticmethod
    def _is_datetime_col(series: pd.Series) -> bool:
        """Check if series is datetime or can be parsed as datetime."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True

        # Handle numeric timestamps (Unix seconds or milliseconds)
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if not non_null.empty:
                val_min, val_max = non_null.min(), non_null.max()
                if (946684800 <= val_min and val_max <= 2147483647) or (
                    946684800000 <= val_min and val_max <= 2147483647000
                ):
                    return True
            return False

        if pd.api.types.is_string_dtype(series) or series.dtype == "object":
            non_null = series.dropna().astype(str).str.strip()
            if non_null.empty:
                return False

            sample = non_null.head(50)
            if sample.empty:
                return False

            date_patterns = [
                r"^\d{4}-\d{2}-\d{2}",
                r"^\d{1,2}/\d{1,2}/\d{4}",
                r"^\d{1,2}-\d{1,2}-\d{4}",
                r"^\d{4}/\d{2}/\d{2}",
                r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}",
            ]
            matches = sum(
                any(re.match(pat, val) for pat in date_patterns) for val in sample
            )
            if matches / len(sample) >= 0.7:
                return True

            try:
                pd.to_datetime(sample, format="mixed", errors="raise")
                return True
            except Exception:
                pass

        return False

    @staticmethod
    def _is_boolean_col(series: pd.Series) -> bool:
        """Check if series is boolean or binary flag."""
        if pd.api.types.is_bool_dtype(series):
            return True
        non_null = series.dropna()
        if non_null.empty:
            return False
        unique_vals = set(non_null.astype(str).str.strip().str.lower().unique())
        boolean_pairs = [
            {"true", "false"},
            {"0", "1"},
            {"0.0", "1.0"},
            {"yes", "no"},
            {"y", "n"},
            {"active", "inactive"},
            {"t", "f"},
        ]
        return any(unique_vals.issubset(pair) for pair in boolean_pairs)

    @staticmethod
    def profile(df: pd.DataFrame) -> dict[str, Any]:
        """Infer column types, per-column statistics, sample values, and quality metrics."""
        row_count = len(df)
        column_count = len(df.columns)

        if row_count == 0:
            return {
                "row_count": 0,
                "column_count": column_count,
                "columns": {},
                "datetime_cols": [],
                "numeric_cols": [],
                "categorical_cols": [],
                "id_cols": [],
                "text_cols": [],
                "boolean_cols": [],
                "duplicate_row_count": 0,
                "data_quality_score": 100.0,
            }

        duplicate_row_count = int(df.duplicated().sum())

        datetime_cols: list[str] = []
        numeric_cols: list[str] = []
        categorical_cols: list[str] = []
        id_cols: list[str] = []
        text_cols: list[str] = []
        boolean_cols: list[str] = []

        columns_meta: dict[str, dict[str, Any]] = {}
        missing_pct_map: dict[str, float] = {}

        for col in df.columns:
            col_str = str(col)
            series = df[col]
            non_null = series.dropna()
            null_count = int(series.isnull().sum())
            missing_pct = round((null_count / row_count) * 100.0, 2)
            missing_pct_map[col_str] = missing_pct

            unique_count = int(non_null.nunique())
            unique_ratio = unique_count / max(row_count, 1)

            # Sample 3 values
            samples = [
                str(v) if pd.notna(v) else "" for v in non_null.head(3).tolist()
            ]

            # Type classification logic
            col_type = "categorical"
            col_lower = col_str.lower()

            is_id_name = (
                col_lower.endswith("_id")
                or col_lower == "id"
                or col_lower.endswith(" id")
                or "uuid" in col_lower
                or "guid" in col_lower
                or "hash" in col_lower
            )

            if is_id_name and unique_count >= min(5, int(0.3 * row_count)):
                col_type = "id"
                id_cols.append(col_str)
            elif DatasetProfiler._is_datetime_col(series):
                col_type = "datetime"
                datetime_cols.append(col_str)
            elif DatasetProfiler._is_boolean_col(series):
                col_type = "boolean"
                boolean_cols.append(col_str)
            elif pd.api.types.is_numeric_dtype(series):
                col_type = "numeric"
                numeric_cols.append(col_str)
            elif unique_ratio >= 0.85 and row_count > 10:
                col_type = "id"
                id_cols.append(col_str)
            elif pd.api.types.is_string_dtype(series) or series.dtype == "object":
                avg_len = (
                    non_null.astype(str).str.len().mean() if not non_null.empty else 0
                )
                if unique_ratio > 0.5 and avg_len > 25:
                    col_type = "text"
                    text_cols.append(col_str)
                else:
                    col_type = "categorical"
                    categorical_cols.append(col_str)

            # Compute stats
            stats: dict[str, Any] = {}
            if col_type == "numeric":
                if not non_null.empty:
                    stats = {
                        "mean": float(np.round(non_null.mean(), 4)),
                        "median": float(np.round(non_null.median(), 4)),
                        "std": float(np.round(non_null.std(ddof=1) if len(non_null) > 1 else 0.0, 4)),
                        "min": float(np.round(non_null.min(), 4)),
                        "max": float(np.round(non_null.max(), 4)),
                        "q1": float(np.round(non_null.quantile(0.25), 4)),
                        "q3": float(np.round(non_null.quantile(0.75), 4)),
                    }
                else:
                    stats = {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "q1": 0.0, "q3": 0.0}
            elif col_type in ["categorical", "text", "boolean", "id"]:
                top_5 = non_null.value_counts().head(5).index.astype(str).tolist() if not non_null.empty else []
                mode_val = str(non_null.mode().iloc[0]) if not non_null.empty and not non_null.mode().empty else ""
                stats = {
                    "top_5": top_5,
                    "mode": mode_val,
                }

            columns_meta[col_str] = {
                "type": col_type,
                "missing_pct": missing_pct,
                "unique_count": unique_count,
                "sample_values": samples,
                "stats": stats,
            }

        # Data Quality Score calculation
        avg_missing = np.mean(list(missing_pct_map.values())) if missing_pct_map else 0.0
        dupe_ratio = (duplicate_row_count / row_count) * 100.0 if row_count > 0 else 0.0
        empty_60_cols = sum(1 for pct in missing_pct_map.values() if pct >= 60.0)

        quality_score = 100.0
        quality_score -= avg_missing * 0.4
        quality_score -= dupe_ratio * 0.3
        quality_score -= empty_60_cols * 3.0
        quality_score = max(0.0, min(100.0, float(np.round(quality_score, 1))))

        return {
            "row_count": row_count,
            "column_count": column_count,
            "columns": columns_meta,
            "datetime_cols": datetime_cols,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "id_cols": id_cols,
            "text_cols": text_cols,
            "boolean_cols": boolean_cols,
            "duplicate_row_count": duplicate_row_count,
            "data_quality_score": quality_score,
            "missing_pct": missing_pct_map,
        }
