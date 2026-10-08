"""Unit tests for the Auto-Analysis Report Engine."""

import pandas as pd
import pytest

from app.analysis.analysis_planner import AnalysisPlanner
from app.analysis.analysis_runner import AnalysisRunner
from app.analysis.dataset_classifier import DatasetClassifier
from app.analysis.dataset_profiler import DatasetProfiler
from app.analysis.pipeline import AutoAnalysisPipeline
from app.analysis.report_builder import ReportBuilder


@pytest.fixture
def sample_subscription_df():
    return pd.DataFrame(
        {
            "customer_id": [f"CUST_{i}" for i in range(100)],
            "signup_date": pd.date_range("2024-01-01", periods=100, freq="D").strftime("%Y-%m-%d"),
            "cancellation_date": [
                (pd.Timestamp("2024-01-01") + pd.Timedelta(days=i * 2)).strftime("%Y-%m-%d")
                if i % 3 == 0
                else None
                for i in range(100)
            ],
            "status": ["active" if i % 3 != 0 else "canceled" for i in range(100)],
            "mrr": [49.0 + (i % 5) * 10 for i in range(100)],
            "plan": ["Basic" if i % 2 == 0 else "Pro" for i in range(100)],
        }
    )


@pytest.fixture
def sample_generic_df():
    return pd.DataFrame(
        {
            "sensor_1": [10.5, 12.1, 11.8, 100.2, 11.0, 12.5, 11.2, 10.9, 11.4, 12.0],
            "sensor_2": [5.1, 5.2, 5.0, 5.5, 5.3, 5.2, 5.1, 5.4, 5.2, 5.3],
            "category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        }
    )


def test_dataset_profiler(sample_subscription_df):
    profile = DatasetProfiler.profile(sample_subscription_df)
    assert profile["row_count"] == 100
    assert profile["column_count"] == 6
    assert "customer_id" in profile["id_cols"]
    assert "signup_date" in profile["datetime_cols"]
    assert profile["data_quality_score"] > 80.0


def test_dataset_classifier(sample_subscription_df):
    profile = DatasetProfiler.profile(sample_subscription_df)
    classifier = DatasetClassifier()
    classification = classifier.classify(sample_subscription_df, profile)
    assert classification["dataset_type"] == "subscription"
    assert classification["key_entity"] == "customer"


def test_analysis_planner(sample_subscription_df):
    profile = DatasetProfiler.profile(sample_subscription_df)
    classifier = DatasetClassifier()
    classification = classifier.classify(sample_subscription_df, profile)
    tasks = AnalysisPlanner.plan(profile, classification)
    assert len(tasks) > 5
    task_names = [t.name for t in tasks]
    assert "descriptive_stats" in task_names
    assert "outlier_detection" in task_names


def test_pipeline_run(sample_subscription_df, tmp_path):
    duckdb_file = str(tmp_path / "test.duckdb")
    pipeline = AutoAnalysisPipeline(duckdb_path=duckdb_file)
    report = pipeline.run(sample_subscription_df, table_name="test_subs")

    assert report.title != ""
    assert report.dataset_type == "subscription"
    assert len(report.sections) > 0
    assert len(report.recommendations) > 0
    assert "Total Records" in report.key_metrics


def test_classifier_robust_json_parsing():
    classifier = DatasetClassifier()
    # Test fallback classification on arbitrary text or empty profile
    res = classifier._fallback_classify(pd.DataFrame({"a": [1, 2, 3]}), {})
    assert res["dataset_type"] == "generic"
    assert res["key_entity"] == "record"
