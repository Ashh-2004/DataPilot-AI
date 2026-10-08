"""Dataset classifier inferring domain type, key entity, and semantic column roles using Ollama with rule fallbacks."""

import json
import logging
import os
import re
from typing import Any

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)


class DatasetClassifier:
    """Classifies dataset into domain categories (subscription, sales, hr, finance, healthcare, etc.) and infers semantic roles."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def _fallback_classify(self, df: pd.DataFrame, profile: dict[str, Any]) -> dict[str, Any]:
        """Heuristic rule-based classification fallback when LLM is unreachable or returns low confidence."""
        columns = [str(c) for c in df.columns]
        columns_lower = [c.lower() for c in columns]
        col_types = profile.get("columns", {})
        dt_cols = profile.get("datetime_cols", [])
        num_cols = profile.get("numeric_cols", [])
        id_cols = profile.get("id_cols", [])
        cat_cols = profile.get("categorical_cols", [])

        id_column = id_cols[0] if id_cols else (columns[0] if columns else None)
        date_columns = dt_cols
        value_columns = [
            c for c in num_cols if any(kw in c.lower() for kw in ["amount", "price", "revenue", "mrr", "total", "score", "qty", "val", "fee", "cost", "salary"])
        ]
        status_column = next(
            (c for c in columns if any(kw in c.lower() for kw in ["status", "state", "churn", "flag", "active", "pass", "outcome"])), None
        )
        group_columns = cat_cols[:3]

        col_str = " ".join(columns_lower)

        # Domain heuristic detection
        if any(kw in col_str for kw in ["subscription", "sub_id", "signup", "cancel", "churn", "mrr", "plan", "recurring"]):
            return {
                "dataset_type": "subscription",
                "key_entity": "customer",
                "id_column": id_column,
                "date_columns": date_columns,
                "value_columns": value_columns,
                "status_column": status_column,
                "group_columns": group_columns,
                "confidence": 0.85,
                "reasoning": "Detected subscription and SaaS retention keywords in column names.",
            }

        if any(kw in col_str for kw in ["order", "revenue", "sales", "price", "product", "sku", "item", "discount"]):
            return {
                "dataset_type": "sales",
                "key_entity": "order",
                "id_column": id_column,
                "date_columns": date_columns,
                "value_columns": value_columns,
                "status_column": status_column,
                "group_columns": group_columns,
                "confidence": 0.85,
                "reasoning": "Detected sales, orders, and transactional commerce fields.",
            }

        if any(kw in col_str for kw in ["patient", "diagnosis", "hospital", "doctor", "admission", "treatment", "medical"]):
            return {
                "dataset_type": "healthcare",
                "key_entity": "patient",
                "id_column": id_column,
                "date_columns": date_columns,
                "value_columns": value_columns,
                "status_column": status_column,
                "group_columns": group_columns,
                "confidence": 0.80,
                "reasoning": "Detected healthcare, medical, or clinical record identifiers.",
            }

        if any(kw in col_str for kw in ["employee", "department", "salary", "hire_date", "tenure", "hr", "payroll"]):
            return {
                "dataset_type": "hr",
                "key_entity": "employee",
                "id_column": id_column,
                "date_columns": date_columns,
                "value_columns": value_columns,
                "status_column": status_column,
                "group_columns": group_columns,
                "confidence": 0.80,
                "reasoning": "Detected human resources, department, and salary attributes.",
            }

        if any(kw in col_str for kw in ["student", "grade", "score", "subject", "exam", "course", "gpa", "school"]):
            return {
                "dataset_type": "education",
                "key_entity": "student",
                "id_column": id_column,
                "date_columns": date_columns,
                "value_columns": value_columns,
                "status_column": status_column,
                "group_columns": group_columns,
                "confidence": 0.80,
                "reasoning": "Detected academic, student performance, or educational course indicators.",
            }

        if any(kw in col_str for kw in ["player", "match", "team", "score", "wickets", "runs", "goals", "league"]):
            return {
                "dataset_type": "sports",
                "key_entity": "match",
                "id_column": id_column,
                "date_columns": date_columns,
                "value_columns": value_columns,
                "status_column": status_column,
                "group_columns": group_columns,
                "confidence": 0.80,
                "reasoning": "Detected sports metrics, player/team scores, or match statistics.",
            }

        if any(kw in col_str for kw in ["sensor", "temperature", "voltage", "reading", "device", "iot", "humidity"]):
            return {
                "dataset_type": "sensor_iot",
                "key_entity": "sensor_reading",
                "id_column": id_column,
                "date_columns": date_columns,
                "value_columns": value_columns,
                "status_column": status_column,
                "group_columns": group_columns,
                "confidence": 0.80,
                "reasoning": "Detected IoT telemetry or environmental sensor reading signals.",
            }

        return {
            "dataset_type": "generic",
            "key_entity": "record",
            "id_column": id_column,
            "date_columns": date_columns,
            "value_columns": value_columns,
            "status_column": status_column,
            "group_columns": group_columns,
            "confidence": 0.60,
            "reasoning": "Domain ambiguous or general tabular dataset; using generic analysis tasks.",
        }

    def classify(self, df: pd.DataFrame, profile: dict[str, Any]) -> dict[str, Any]:
        """Classify dataset using Ollama LLM, enforcing valid JSON schema and confidence check."""
        columns = [str(c) for c in df.columns]
        sample_records = df.head(3).to_dict(orient="records")

        prompt = (
            f"Given column names: {columns}, sample rows: {sample_records},\n"
            "infer the following. Respond ONLY in valid JSON:\n"
            "{\n"
            '  "dataset_type": "subscription" | "sales" | "hr" | "finance" | "healthcare" | "logistics" | "marketing" | "education" | "sports" | "real_estate" | "sensor_iot" | "generic",\n'
            '  "key_entity": "customer" | "employee" | "order" | "patient" | "student" | "match" | "sensor_reading" | "record",\n'
            '  "id_column": "col_name or null",\n'
            '  "date_columns": ["col_name"] or [],\n'
            '  "value_columns": ["col_name"] or [],\n'
            '  "status_column": "col_name or null",\n'
            '  "group_columns": ["col_name"] or [],\n'
            '  "confidence": float_0_to_1,\n'
            '  "reasoning": "one sentence explanation"\n'
            "}\n"
            "Do NOT include any commentary outside valid JSON."
        )

        try:
            res = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 256},
                },
                timeout=35,
            )
            if res.status_code == 200:
                body = res.json()
                raw_text = body.get("response", "").strip()

                # Extract content between first '{' and last '}'
                first_brace = raw_text.find("{")
                last_brace = raw_text.rfind("}")
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    clean_json = raw_text[first_brace : last_brace + 1]
                    parsed = json.loads(clean_json)
                    if isinstance(parsed, dict) and "dataset_type" in parsed:
                        confidence = float(parsed.get("confidence", 0.0))
                        if confidence >= 0.5:
                            return parsed
                        else:
                            LOGGER.info("Classifier LLM confidence %.2f < 0.5; falling back", confidence)
        except Exception as exc:
            LOGGER.warning("Ollama classification call failed (%s); using fallback", exc)

        try:
            return self._fallback_classify(df, profile)
        except Exception as exc:
            LOGGER.warning("Fallback classifier failed (%s); returning generic default", exc)
            return {
                "dataset_type": "generic",
                "key_entity": "record",
                "id_column": None,
                "date_columns": [],
                "value_columns": [],
                "status_column": None,
                "group_columns": [],
                "confidence": 0.0,
                "reasoning": "Classification failed, using generic analyzer",
            }
