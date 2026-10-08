"""Dynamic analysis planner triggering universal and domain-specific tasks based on dataset profiling."""

import logging
from typing import Any

from app.analysis.dataclasses import AnalysisTask

LOGGER = logging.getLogger(__name__)


class AnalysisPlanner:
    """Plans dynamic execution sequence of AnalysisTasks based on existing column types and inferred dataset type."""

    @staticmethod
    def plan(profile: dict[str, Any], classification: dict[str, Any]) -> list[AnalysisTask]:
        """Generate list of AnalysisTasks tailored to available dataset columns and domain classification."""
        from app.analysis import analysis_runner as runner

        tasks: list[AnalysisTask] = []

        num_cols = profile.get("numeric_cols", [])
        cat_cols = profile.get("categorical_cols", [])
        dt_cols = profile.get("datetime_cols", [])
        bool_cols = profile.get("boolean_cols", [])
        id_cols = profile.get("id_cols", [])

        id_col = classification.get("id_column") or (id_cols[0] if id_cols else None)
        date_col = (classification.get("date_columns") or dt_cols or [None])[0]
        status_col = classification.get("status_column") or (bool_cols[0] if bool_cols else None)
        val_cols = classification.get("value_columns") or num_cols

        # -------------------------------------------------------------------
        # Universal Column-Condition Tasks
        # -------------------------------------------------------------------

        # 1. Any numeric columns -> Descriptive stats
        if num_cols:
            tasks.append(
                AnalysisTask(
                    name="descriptive_stats",
                    description="Compute mean, median, std, min, max, Q1, Q3 and identify highest variance/skewed numeric columns.",
                    required_columns=num_cols,
                    analysis_fn=runner.descriptive_stats,
                )
            )

        # 2. Any numeric columns -> Distribution analysis
        if num_cols:
            tasks.append(
                AnalysisTask(
                    name="distribution_analysis",
                    description="Examine histograms, skewness, kurtosis, and normality flags for numeric columns.",
                    required_columns=num_cols,
                    analysis_fn=runner.distribution_analysis,
                )
            )

        # 3. 2+ numeric columns -> Correlation matrix
        if len(num_cols) >= 2:
            tasks.append(
                AnalysisTask(
                    name="correlation_analysis",
                    description="Compute Pearson correlation matrix and identify top positive and negative co-dependencies.",
                    required_columns=num_cols,
                    analysis_fn=runner.correlation_analysis,
                )
            )

        # 4. Any numeric columns -> Outlier detection (IsolationForest + SHAP)
        if num_cols:
            tasks.append(
                AnalysisTask(
                    name="outlier_detection",
                    description="Detect statistical anomalies with IsolationForest and explain driving attributes using SHAP values.",
                    required_columns=num_cols,
                    analysis_fn=runner.outlier_detection,
                )
            )

        # 5. Any categorical columns -> Value frequency analysis
        if cat_cols:
            tasks.append(
                AnalysisTask(
                    name="value_frequency",
                    description="Analyze category frequencies, top values, and dominant classes.",
                    required_columns=cat_cols,
                    analysis_fn=runner.value_frequency,
                )
            )

        # 6. 1+ datetime column -> Time trend analysis
        if date_col:
            tasks.append(
                AnalysisTask(
                    name="time_trend_analysis",
                    description="Resample values by month/week to identify peak periods and overall trajectory.",
                    required_columns=[date_col],
                    analysis_fn=runner.time_trend_analysis,
                )
            )

        # 7. status or boolean column -> Status breakdown analysis
        if status_col:
            tasks.append(
                AnalysisTask(
                    name="status_breakdown",
                    description="Analyze status distributions, dominant state ratios, and state frequencies.",
                    required_columns=[status_col],
                    analysis_fn=runner.status_breakdown,
                )
            )

        # 8. id column + datetime column -> Cohort / retention analysis
        if id_col and date_col:
            tasks.append(
                AnalysisTask(
                    name="cohort_retention",
                    description="Track signup cohorts and calculate D7/D30/D60/D90 retention survival curves.",
                    required_columns=[id_col, date_col],
                    analysis_fn=runner.cohort_retention,
                )
            )

        # 9. id column + numeric columns -> Entity-level aggregation
        if id_col and num_cols:
            tasks.append(
                AnalysisTask(
                    name="entity_aggregation",
                    description="Aggregate activity metrics per unique entity ID to inspect activity distributions.",
                    required_columns=[id_col] + num_cols[:2],
                    analysis_fn=runner.entity_aggregation,
                )
            )

        # 10. categorical + numeric -> Group comparison
        if cat_cols and num_cols:
            tasks.append(
                AnalysisTask(
                    name="group_comparison",
                    description="Compare mean and median numeric metrics across primary categorical groups.",
                    required_columns=[cat_cols[0], num_cols[0]],
                    analysis_fn=runner.group_comparison,
                )
            )

        # 11. 2+ categorical columns -> Cross-tabulation
        if len(cat_cols) >= 2:
            tasks.append(
                AnalysisTask(
                    name="cross_tabulation",
                    description="Compute normalized cross-tabulations between primary categorical features.",
                    required_columns=cat_cols[:2],
                    analysis_fn=runner.cross_tabulation,
                )
            )

        # -------------------------------------------------------------------
        # Domain-Specific Additional Tasks
        # -------------------------------------------------------------------
        domain_type = classification.get("dataset_type", "generic").lower()

        if domain_type == "subscription":
            tasks.append(
                AnalysisTask(
                    name="domain_subscription_kpis",
                    description="Evaluate subscription churn, implied MRR, paid/unpaid splits, and lifetime retention.",
                    required_columns=dt_cols + num_cols,
                    analysis_fn=runner.domain_subscription_kpis,
                )
            )
        elif domain_type == "sales":
            tasks.append(
                AnalysisTask(
                    name="domain_sales_kpis",
                    description="Evaluate AOV, repeat purchase rates, product top SKUs, and Pareto revenue concentration.",
                    required_columns=num_cols + cat_cols,
                    analysis_fn=runner.domain_sales_kpis,
                )
            )
        elif domain_type == "hr":
            tasks.append(
                AnalysisTask(
                    name="domain_hr_kpis",
                    description="Evaluate department headcount breakdowns, tenure distributions, and salary bands.",
                    required_columns=num_cols + cat_cols,
                    analysis_fn=runner.domain_hr_kpis,
                )
            )
        elif domain_type == "healthcare":
            tasks.append(
                AnalysisTask(
                    name="domain_healthcare_kpis",
                    description="Evaluate diagnosis frequencies, length of stay, and patient admission trends.",
                    required_columns=num_cols + cat_cols,
                    analysis_fn=runner.domain_healthcare_kpis,
                )
            )
        elif domain_type == "marketing":
            tasks.append(
                AnalysisTask(
                    name="domain_marketing_kpis",
                    description="Evaluate acquisition channels, conversion rates, and campaign efficiency.",
                    required_columns=num_cols + cat_cols,
                    analysis_fn=runner.domain_marketing_kpis,
                )
            )
        elif domain_type == "education":
            tasks.append(
                AnalysisTask(
                    name="domain_education_kpis",
                    description="Evaluate test score distributions, pass/fail ratios, and subject comparisons.",
                    required_columns=num_cols + cat_cols,
                    analysis_fn=runner.domain_education_kpis,
                )
            )
        elif domain_type == "sports":
            tasks.append(
                AnalysisTask(
                    name="domain_sports_kpis",
                    description="Evaluate player/team scoring performance, win/loss rates, and match statistics.",
                    required_columns=num_cols + cat_cols,
                    analysis_fn=runner.domain_sports_kpis,
                )
            )
        elif domain_type == "sensor_iot":
            tasks.append(
                AnalysisTask(
                    name="domain_sensor_kpis",
                    description="Evaluate signal stability, telemetry spike detection, and device location readings.",
                    required_columns=num_cols + dt_cols,
                    analysis_fn=runner.domain_sensor_kpis,
                )
            )

        return tasks
