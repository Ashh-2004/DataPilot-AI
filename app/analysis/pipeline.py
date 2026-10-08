"""Auto-analysis pipeline orchestrating data cleaning, profiling, classification, dynamic planning, execution, report assembly, DuckDB loading, and Redis caching."""

import json
import logging
import os
from typing import Any

import duckdb
import pandas as pd
import redis

from app.analysis.analysis_planner import AnalysisPlanner
from app.analysis.analysis_runner import AnalysisRunner
from app.analysis.dataclasses import Report
from app.analysis.dataset_classifier import DatasetClassifier
from app.analysis.dataset_profiler import DatasetProfiler
from app.analysis.report_builder import ReportBuilder
from app.services.cleaning import DataCleaner

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("datapilot.pipeline")

# In-memory cache fallback for Redis-less environments
MEMORY_CACHE: dict[str, Any] = {}


class AutoAnalysisPipeline:
    """Orchestrates end-to-end dataset auto-analysis report generation and database seeding."""

    def __init__(self, duckdb_path: str | None = None, redis_host: str | None = None, redis_port: int | None = None) -> None:
        self.duckdb_path = duckdb_path or os.getenv("DUCKDB_PATH", "./data/datapilot.duckdb")
        self.redis_host = redis_host or os.getenv("REDIS_HOST", "localhost")
        self.redis_port = redis_port or int(os.getenv("REDIS_PORT", 6379))

    def run(self, df: pd.DataFrame, table_name: str = "user_data") -> Report:
        """Run complete auto-analysis pipeline with detailed step logging and Redis state persistence."""
        logger.debug("Step 1: Starting DataCleaner.clean for %d rows...", len(df))
        df_clean = DataCleaner.clean(df)

        logger.debug("Step 2: Starting DatasetProfiler.profile...")
        profile = DatasetProfiler.profile(df_clean)

        logger.debug("Step 3: Starting DatasetClassifier.classify...")
        try:
            classifier = DatasetClassifier()
            classification = classifier.classify(df_clean, profile)
        except Exception as exc:
            logger.warning("Classification failed (%s); using generic fallback", exc)
            classification = {
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

        logger.debug("Step 4: Persisting profile, classification, and sample to Redis...")
        sample_dict = df_clean.head(5).to_dict(orient="records")
        sample_json = df_clean.head(5).to_json()

        # Update in-memory fallback
        MEMORY_CACHE["dataset_profile"] = profile
        MEMORY_CACHE["dataset_classification"] = classification
        MEMORY_CACHE["data_sample"] = sample_dict

        try:
            r = redis.Redis(host=self.redis_host, port=self.redis_port, db=0, socket_connect_timeout=2)
            r.set("dataset_profile", json.dumps(profile))
            r.set("dataset_classification", json.dumps(classification))
            r.set("data_sample", sample_json)
            logger.info("Successfully persisted dataset_profile, dataset_classification, and data_sample to Redis")
        except Exception as exc:
            logger.warning("Redis persistence failed (%s); relying on in-memory cache", exc)

        logger.debug("Step 5: Starting AnalysisPlanner.plan...")
        tasks = AnalysisPlanner.plan(profile, classification)

        logger.debug("Step 6: Starting AnalysisRunner.run_all for %d tasks...", len(tasks))
        results = AnalysisRunner.run_all(df_clean, tasks, profile, classification)

        logger.debug("Step 7: Starting ReportBuilder.build...")
        builder = ReportBuilder()
        report = builder.build(results, classification, profile)

        logger.debug("Step 8: Loading clean dataset into DuckDB table '%s'...", table_name)
        try:
            self._load_into_duckdb(df_clean, table_name)
        except Exception as exc:
            logger.error("Fatal: Failed to load clean dataset into DuckDB: %s", exc)
            raise RuntimeError(f"DuckDB database load failed: {exc}") from exc

        logger.debug("Step 9: Caching Report to Redis key 'latest_report'...")
        self._cache_report(report)

        logger.info("AutoAnalysisPipeline completed successfully. Report title: '%s'", report.title)
        return report

    def _load_into_duckdb(self, df: pd.DataFrame, table_name: str) -> None:
        """Load clean DataFrame into DuckDB as user_data and specified table_name."""
        os.makedirs(os.path.dirname(self.duckdb_path), exist_ok=True)
        with duckdb.connect(self.duckdb_path) as conn:
            conn.register("df_temp", df)
            conn.execute("CREATE OR REPLACE TABLE user_data AS SELECT * FROM df_temp")
            if table_name != "user_data":
                conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df_temp")

    def _cache_report(self, report: Report) -> None:
        """Store Report dictionary in Redis with fallback to in-memory dictionary."""
        report_dict = report.to_dict()
        MEMORY_CACHE["latest_report"] = report_dict
        try:
            r = redis.Redis(host=self.redis_host, port=self.redis_port, db=0, socket_connect_timeout=2)
            r.set("latest_report", json.dumps(report_dict))
            logger.info("Successfully cached report in Redis key 'latest_report'")
        except Exception as exc:
            logger.warning("Redis cache unavailable (%s); stored in in-memory cache", exc)
