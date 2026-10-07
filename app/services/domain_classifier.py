"""Domain classification service using Ollama LLM with strict JSON validation and keyword fallback."""

import json
import logging
import re
from typing import Any

from langchain_ollama import ChatOllama

LOGGER = logging.getLogger(__name__)

SUPPORTED_DOMAINS = [
    "Sales",
    "HR",
    "Healthcare",
    "Finance",
    "Marketing",
    "Inventory",
    "Education",
    "Other",
]


def classify_domain(
    columns: list[str],
    dtypes: dict[str, str],
    sample_rows: list[dict[str, Any]],
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
) -> dict[str, Any]:
    """Classify dataset domain based ONLY on column names, dtypes, and <= 5 sample rows.

    Returns
    -------
    dict with strict keys:
    domain, confidence, reason, likely_kpis, likely_dimensions, suggested_questions (5 items).
    """
    truncated_samples = _truncate_sample_rows(sample_rows[:5])
    col_summary = [f"{col} ({dtypes.get(col, 'UNKNOWN')})" for col in columns]

    prompt_text = (
        "You are an expert Data Architect. Classify the business domain of a dataset based ONLY on its schema and sample rows.\n\n"
        f"Columns & Types: {col_summary}\n"
        f"Sample Rows (<=5): {truncated_samples}\n\n"
        "Return ONLY a valid, single JSON object with no markdown codeblocks, matching this EXACT schema:\n"
        "{\n"
        '  "domain": "Sales | HR | Healthcare | Finance | Marketing | Inventory | Education | Other",\n'
        '  "confidence": 0.95,\n'
        '  "reason": "Brief explanation",\n'
        '  "likely_kpis": ["kpi1", "kpi2"],\n'
        '  "likely_dimensions": ["dim1", "dim2"],\n'
        '  "suggested_questions": ["Q1", "Q2", "Q3", "Q4", "Q5"]\n'
        "}\n\n"
        "CRITICAL: 'suggested_questions' MUST contain EXACTLY 5 distinct analytical questions tailored to this dataset."
    )

    # Attempt 1: Standard LLM call
    llm = ChatOllama(model=model, base_url=base_url, temperature=0.1, num_ctx=2048, num_predict=500)
    try:
        response = llm.invoke(prompt_text)
        content = response.content if isinstance(response.content, str) else str(response.content)
        parsed = _parse_and_validate_json(content)
        if parsed:
            return parsed
    except Exception as exc:
        LOGGER.warning("Domain classifier Attempt 1 failed: %s", exc)

    # Attempt 2: Strict retry prompt
    retry_prompt = (
        "CRITICAL SYSTEM INSTRUCTION: Output ONLY raw valid JSON. Do not include markdown tags like ```json.\n\n"
        + prompt_text
    )
    try:
        response = llm.invoke(retry_prompt)
        content = response.content if isinstance(response.content, str) else str(response.content)
        parsed = _parse_and_validate_json(content)
        if parsed:
            return parsed
    except Exception as exc:
        LOGGER.warning("Domain classifier Attempt 2 failed: %s", exc)

    # Fallback to keyword heuristic
    LOGGER.info("Domain classifier falling back to keyword heuristic")
    return _fallback_domain_heuristic(columns, dtypes)


def _truncate_sample_rows(sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Truncate long string values in sample rows to keep prompt token footprint minimal."""
    truncated: list[dict[str, Any]] = []
    for row in sample_rows[:5]:
        new_row: dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, str) and len(v) > 50:
                new_row[k] = v[:47] + "..."
            else:
                new_row[k] = v
        truncated.append(new_row)
    return truncated


def _parse_and_validate_json(content: str) -> dict[str, Any] | None:
    """Parse JSON string and validate required domain classification schema."""
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        return None

    try:
        data = json.loads(content[start : end + 1])
        if not isinstance(data, dict):
            return None

        domain = str(data.get("domain", "")).strip().title()
        if not domain:
            return None

        confidence = float(data.get("confidence", 0.8))
        confidence = min(1.0, max(0.0, confidence))

        reason = str(data.get("reason", "Inferred from dataset schema and sample attributes."))
        kpis = [str(x) for x in data.get("likely_kpis", []) if x][:5]
        dimensions = [str(x) for x in data.get("likely_dimensions", []) if x][:5]

        raw_questions = [str(q).strip() for q in data.get("suggested_questions", []) if q]
        questions = list(dict.fromkeys(raw_questions))

        # Pad to exactly 5 suggested questions if necessary
        fallback_qs = [
            "What are the top summary statistics across key metrics?",
            "How do main metrics break down by primary categories?",
            "Are there any significant trends over time?",
            "What are the key outliers or unusual observations?",
            "Which dimensions contribute most to overall performance?",
        ]
        for fq in fallback_qs:
            if len(questions) < 5 and fq not in questions:
                questions.append(fq)

        return {
            "domain": domain,
            "confidence": confidence,
            "reason": reason,
            "likely_kpis": kpis,
            "likely_dimensions": dimensions,
            "suggested_questions": questions[:5],
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def _fallback_domain_heuristic(columns: list[str], dtypes: dict[str, str]) -> dict[str, Any]:
    """Deterministic keyword fallback classifier."""
    col_str = (" ".join(columns) + " " + " ".join(dtypes.keys())).lower()

    if any(k in col_str for k in ("sale", "order", "product", "customer", "revenue", "profit", "store", "unit_price")):
        domain = "Sales"
        kpis = ["total_revenue", "order_count", "average_order_value"]
        dims = ["product_category", "customer_segment", "store_region"]
    elif any(k in col_str for k in ("employee", "salary", "department", "hire", "hr", "payroll", "job", "worker")):
        domain = "HR"
        kpis = ["headcount", "average_salary", "tenure_years"]
        dims = ["department", "job_role", "office_location"]
    elif any(k in col_str for k in ("patient", "hospital", "clinical", "diagnosis", "disease", "treatment", "doctor", "comorbidity", "pneumonia")):
        domain = "Healthcare"
        kpis = ["patient_count", "readmission_rate", "average_stay_days"]
        dims = ["diagnosis_category", "patient_type", "treatment_group"]
    elif any(k in col_str for k in ("transaction", "account", "balance", "bank", "credit", "loan", "debit", "interest")):
        domain = "Finance"
        kpis = ["total_balance", "transaction_volume", "default_rate"]
        dims = ["account_type", "credit_tier", "transaction_channel"]
    elif any(k in col_str for k in ("campaign", "click", "impression", "lead", "conversion", "ad", "channel")):
        domain = "Marketing"
        kpis = ["conversion_rate", "cost_per_acquisition", "total_clicks"]
        dims = ["campaign_name", "traffic_source", "ad_group"]
    elif any(k in col_str for k in ("stock", "warehouse", "sku", "supplier", "inventory", "reorder", "quantity")):
        domain = "Inventory"
        kpis = ["total_stock_value", "turnover_rate", "out_of_stock_count"]
        dims = ["warehouse_location", "supplier_name", "item_category"]
    elif any(k in col_str for k in ("student", "course", "grade", "gpa", "exam", "school", "teacher", "enrollment")):
        domain = "Education"
        kpis = ["average_gpa", "pass_rate", "student_count"]
        dims = ["course_name", "department", "academic_year"]
    else:
        domain = "Other"
        kpis = ["row_count", "column_count"]
        dims = [col for col in columns[:3]]

    # Generate 5 questions for fallback domain
    num_cols = [c for c in columns if any(t in dtypes.get(c, "").upper() for t in ("INT", "DOUBLE", "FLOAT", "DECIMAL", "NUMERIC"))]
    cat_cols = [c for c in columns if c not in num_cols]

    n1 = num_cols[0] if num_cols else "metric"
    c1 = cat_cols[0] if cat_cols else "category"

    questions = [
        f"What is the total distribution of {n1} across the dataset?",
        f"Which {c1} accounts for the highest values?",
        f"How do key metrics break down by major dimensions?",
        f"Are there any notable anomalies or outliers in {n1}?",
        f"What are the top 5 records ranked by {n1}?",
    ]

    return {
        "domain": domain,
        "confidence": 0.70,
        "reason": f"Inferred domain '{domain}' via keyword match on schema columns.",
        "likely_kpis": kpis,
        "likely_dimensions": dims,
        "suggested_questions": questions[:5],
    }
