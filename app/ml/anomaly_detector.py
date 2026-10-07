"""Anomaly detection with IsolationForest + SHAP, falling back to Z-score when native DLLs are restricted."""

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)

# Try optional imports
try:
    import shap
    from sklearn.ensemble import IsolationForest
    _SKLEARN_AVAILABLE = True
except Exception as exc:
    LOGGER.warning("sklearn/shap not available (%s); falling back to statistical anomaly detection", exc)
    shap = None
    IsolationForest = None
    _SKLEARN_AVAILABLE = False


class AnomalyDetector:
    """Detect numeric outliers and explain their influential columns."""

    def __init__(self) -> None:
        """Configure the detector from environment variables."""
        load_dotenv()
        self.contamination = float(os.getenv("ANOMALY_CONTAMINATION", "0.05"))
        self.min_rows = int(os.getenv("ANOMALY_MIN_ROWS", "10"))

    def detect(self, df: pd.DataFrame) -> dict[str, Any]:
        """Return flagged rows, anomaly columns, scores, and detection status."""
        empty_result = self._empty_result()
        if len(df) < self.min_rows:
            return empty_result

        numeric_df = df.select_dtypes(include=[np.number]).copy()
        numeric_df = numeric_df.dropna(axis=1, how="all")
        if numeric_df.empty:
            return empty_result

        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
        numeric_df = numeric_df.fillna(numeric_df.median()).fillna(0.0)

        if _SKLEARN_AVAILABLE and IsolationForest is not None:
            try:
                model = IsolationForest(contamination=self.contamination, random_state=42)
                predictions = model.fit_predict(numeric_df)
                anomaly_indices = np.flatnonzero(predictions == -1)
                scores = -model.score_samples(numeric_df)

                explanations = self._shap_explanations(model, numeric_df, anomaly_indices)
                anomaly_columns = sorted({column for columns in explanations.values() for column in columns})
                anomaly_rows = [
                    {
                        **self._json_safe_row(df.iloc[index].to_dict()),
                        "_anomaly_score": float(scores[index]),
                    }
                    for index in anomaly_indices
                ]
                return {
                    "anomaly_rows": anomaly_rows,
                    "anomaly_count": int(len(anomaly_indices)),
                    "anomaly_columns": anomaly_columns,
                    "has_anomaly": bool(len(anomaly_indices)),
                    "anomaly_scores": [float(score) for score in scores],
                }
            except Exception:
                LOGGER.warning("IsolationForest failed, falling back to Z-score", exc_info=True)

        # Fallback: Robust Z-score anomaly detection
        return self._statistical_detect(df, numeric_df)

    def _statistical_detect(self, df: pd.DataFrame, numeric_df: pd.DataFrame) -> dict[str, Any]:
        """Pure-numpy Z-score fallback for anomaly detection."""
        means = numeric_df.mean()
        stds = numeric_df.std().replace(0, 1.0)
        z_scores = ((numeric_df - means) / stds).abs()
        max_z = z_scores.max(axis=1)

        # Flag top contamination fraction or z > 2.5
        threshold = max(2.5, float(np.percentile(max_z, 100 * (1 - self.contamination))))
        anomaly_mask = max_z >= threshold
        anomaly_indices = np.flatnonzero(anomaly_mask)

        flagged_cols = set()
        for idx in anomaly_indices:
            row_z = z_scores.iloc[idx]
            top_cols = row_z.nlargest(3).index.tolist()
            flagged_cols.update(top_cols)

        anomaly_rows = [
            {
                **self._json_safe_row(df.iloc[index].to_dict()),
                "_anomaly_score": float(max_z.iloc[index]),
            }
            for index in anomaly_indices
        ]
        return {
            "anomaly_rows": anomaly_rows,
            "anomaly_count": int(len(anomaly_indices)),
            "anomaly_columns": sorted(flagged_cols),
            "has_anomaly": bool(len(anomaly_indices)),
            "anomaly_scores": [float(s) for s in max_z],
        }

    def _shap_explanations(
        self,
        model: Any,
        numeric_df: pd.DataFrame,
        anomaly_indices: np.ndarray,
    ) -> dict[int, list[str]]:
        """Return the three highest-impact numeric columns for each flagged row."""
        if not len(anomaly_indices):
            return {}
        try:
            if shap is not None:
                shap_values = shap.TreeExplainer(model).shap_values(numeric_df)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                shap_array = np.asarray(shap_values)
                return {
                    int(index): [
                        numeric_df.columns[column_index]
                        for column_index in np.argsort(np.abs(shap_array[index]))[-3:][::-1]
                    ]
                    for index in anomaly_indices
                }
        except Exception:
            LOGGER.warning("SHAP explanation failed; using all numeric columns", exc_info=True)
        columns = list(numeric_df.columns)
        return {int(index): columns for index in anomaly_indices}

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        """Return the stable result shape for skipped or clean inputs."""
        return {
            "anomaly_rows": [],
            "anomaly_count": 0,
            "anomaly_columns": [],
            "has_anomaly": False,
            "anomaly_scores": [],
        }

    @staticmethod
    def _json_safe_row(row: dict[str, Any]) -> dict[str, Any]:
        """Convert Pandas and NumPy scalar values into JSON-safe values."""
        safe_row: dict[str, Any] = {}
        for key, value in row.items():
            if pd.isna(value):
                safe_row[key] = None
            elif isinstance(value, np.generic):
                safe_row[key] = value.item()
            else:
                safe_row[key] = value
        return safe_row
