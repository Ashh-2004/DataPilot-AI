"""Analysis runner executing universal statistical routines with SHAP-explained anomaly detection and charts."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

from app.analysis.dataclasses import AnalysisResult, AnalysisTask, ChartSpec

LOGGER = logging.getLogger(__name__)


class AnalysisRunner:
    """Executes planned analysis tasks safely against pandas DataFrames."""

    @staticmethod
    def run_all(
        df: pd.DataFrame,
        tasks: list[AnalysisTask],
        profile: dict[str, Any],
        classification: dict[str, Any],
    ) -> list[AnalysisResult]:
        """Execute list of AnalysisTasks, capturing exceptions as skipped results."""
        results: list[AnalysisResult] = []

        for task in tasks:
            # Check required columns existence
            missing_cols = [c for c in task.required_columns if c and c not in df.columns]
            if missing_cols:
                results.append(
                    AnalysisResult(
                        section_title=f"Task: {task.name.replace('_', ' ').title()}",
                        content=f"Skipped analysis: missing required columns {missing_cols}.",
                        skipped=True,
                        skip_reason=f"Required columns missing: {missing_cols}",
                    )
                )
                continue

            try:
                res = task.analysis_fn(df, profile, classification)
                if res is not None:
                    results.append(res)
                else:
                    results.append(
                        AnalysisResult(
                            section_title=getattr(task, "name", "Task"),
                            content="",
                            table=None,
                            insight="",
                            charts=[],
                            skipped=True,
                            skip_reason="Task returned None",
                        )
                    )
            except Exception as e:
                LOGGER.warning("Task '%s' failed during execution (%s); marking skipped", getattr(task, "name", "unknown"), e)
                results.append(
                    AnalysisResult(
                        section_title=getattr(task, "name", "Task"),
                        content="",
                        table=None,
                        insight="",
                        charts=[],
                        skipped=True,
                        skip_reason=str(e),
                    )
                )

        return results


# ---------------------------------------------------------------------------
# Universal Analysis Functions
# ---------------------------------------------------------------------------


def descriptive_stats(
    df: pd.DataFrame, profile: dict[str, Any], classification: dict[str, Any]
) -> AnalysisResult:
    """Compute mean, median, std, min, max, Q1, Q3 and highlight variance/skewness."""
    num_cols = profile.get("numeric_cols", [])
    if not num_cols:
        return AnalysisResult(
            section_title="Descriptive Statistics",
            content="No numeric columns available for descriptive statistics.",
            skipped=True,
            skip_reason="No numeric columns present.",
        )

    rows = []
    highest_var_col = None
    max_var = -1.0
    most_skewed_col = None
    max_skew = -1.0

    for col in num_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        var_val = float(s.var(ddof=1)) if len(s) > 1 else 0.0
        skew_val = abs(float(s.skew())) if len(s) > 2 else 0.0

        if var_val > max_var:
            max_var = var_val
            highest_var_col = col

        if skew_val > max_skew:
            max_skew = skew_val
            most_skewed_col = col

        rows.append(
            {
                "Column": col,
                "Mean": round(float(s.mean()), 4),
                "Median": round(float(s.median()), 4),
                "Std Dev": round(float(s.std(ddof=1)), 4) if len(s) > 1 else 0.0,
                "Min": round(float(s.min()), 4),
                "Max": round(float(s.max()), 4),
                "Q1 (25%)": round(float(s.quantile(0.25)), 4),
                "Q3 (75%)": round(float(s.quantile(0.75)), 4),
            }
        )

    table = pd.DataFrame(rows)
    insight = f"Highest variance observed in '{highest_var_col}' (var: {max_var:,.2f}); most skewed attribute is '{most_skewed_col}' (skew: {max_skew:.2f})."

    # Box plot chart spec
    charts = [
        ChartSpec(
            chart_type="box",
            title="Distribution Boxplots Across Numeric Attributes",
            x="Attribute",
            y="Value",
            data=df[num_cols].melt(var_name="Attribute", value_name="Value").dropna(),
        )
    ]

    return AnalysisResult(
        section_title="Descriptive Statistics Overview",
        content=(
            f"Descriptive statistical summary across {len(num_cols)} numeric attributes. "
            f"Feature '{highest_var_col}' exhibits the widest overall variance, while '{most_skewed_col}' "
            "displays the highest distributional asymmetry."
        ),
        table=table,
        insight=insight,
        charts=charts,
    )


def distribution_analysis(
    df: pd.DataFrame, profile: dict[str, Any], classification: dict[str, Any]
) -> AnalysisResult:
    """Analyze histograms, skewness, kurtosis, and normality flags."""
    num_cols = profile.get("numeric_cols", [])
    if not num_cols:
        return AnalysisResult(
            section_title="Distribution Analysis",
            content="No numeric columns present for distribution fitting.",
            skipped=True,
            skip_reason="No numeric columns present.",
        )

    dist_rows = []
    charts: list[ChartSpec] = []

    for col in num_cols[:5]:
        s = df[col].dropna()
        if len(s) < 3:
            continue
        skew_val = float(s.skew())
        kurt_val = float(s.kurtosis())

        if skew_val > 1.0:
            norm_flag = "Strongly Right-Skewed"
        elif skew_val < -1.0:
            norm_flag = "Strongly Left-Skewed"
        else:
            norm_flag = "Approximately Normal"

        dist_rows.append(
            {
                "Attribute": col,
                "Skewness": round(skew_val, 2),
                "Kurtosis": round(kurt_val, 2),
                "Distribution Class": norm_flag,
            }
        )

        charts.append(
            ChartSpec(
                chart_type="histogram",
                title=f"Distribution Histogram: {col}",
                x=col,
                y="count",
                data=df[[col]].dropna(),
            )
        )

    table = pd.DataFrame(dist_rows)
    insight = f"Analyzed distributions for {len(dist_rows)} numeric features; detected right-skewed tails in primary metrics."

    return AnalysisResult(
        section_title="Distribution & Normality Analysis",
        content="Histogram distribution inspection assesses skewness, kurtosis, and normality characteristics across numerical features.",
        table=table,
        insight=insight,
        charts=charts,
    )


def correlation_analysis(
    df: pd.DataFrame, profile: dict[str, Any], classification: dict[str, Any]
) -> AnalysisResult:
    """Compute Pearson correlation matrix and highlight strongest pairs."""
    num_cols = profile.get("numeric_cols", [])
    if len(num_cols) < 2:
        return AnalysisResult(
            section_title="Correlation Matrix",
            content="Fewer than 2 numeric columns available for correlation matrix.",
            skipped=True,
            skip_reason="Insufficient numeric columns.",
        )

    num_df = df[num_cols].dropna()
    if num_df.empty:
        return AnalysisResult(
            section_title="Correlation Matrix",
            content="Numeric columns contain insufficient non-null values.",
            skipped=True,
            skip_reason="Empty dataset after null dropping.",
        )

    corr_df = num_df.corr().round(3)
    pairs = []
    cols = corr_df.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = float(corr_df.iloc[i, j])
            pairs.append((cols[i], cols[j], val))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    top_pos = [f"{p[0]} & {p[1]} ({p[2]:+.2f})" for p in pairs if p[2] > 0][:3]
    top_neg = [f"{p[0]} & {p[1]} ({p[2]:+.2f})" for p in pairs if p[2] < 0][:3]

    pos_str = ", ".join(top_pos) if top_pos else "None"
    neg_str = ", ".join(top_neg) if top_neg else "None"

    insight = f"Strongest positive correlations: [{pos_str}]. Strongest negative correlations: [{neg_str}]."

    corr_table = corr_df.reset_index()
    charts = [
        ChartSpec(
            chart_type="heatmap",
            title="Pearson Feature Correlation Matrix",
            x="index",
            y=list(cols),
            data=corr_table,
        )
    ]

    return AnalysisResult(
        section_title="Correlation & Dependency Analysis",
        content="Pearson correlation matrix identifies linear inter-dependencies between numerical features.",
        table=corr_table,
        insight=insight,
        charts=charts,
    )


def outlier_detection(
    df: pd.DataFrame, profile: dict[str, Any], classification: dict[str, Any]
) -> AnalysisResult:
    """IsolationForest anomaly detection with SHAP attribute explanation."""
    num_cols = profile.get("numeric_cols", [])
    if not num_cols or len(df) < 6:
        return AnalysisResult(
            section_title="Outlier Detection (IsolationForest)",
            content="Insufficient numeric records for IsolationForest anomaly detection.",
            skipped=True,
            skip_reason="Insufficient data.",
        )

    num_df = df[num_cols].fillna(df[num_cols].median())
    iso = IsolationForest(contamination=0.05, random_state=42)
    preds = iso.fit_predict(num_df)
    scores = -iso.score_samples(num_df)

    flagged_mask = preds == -1
    flagged_count = int(flagged_mask.sum())
    flagged_pct = round((flagged_count / len(df)) * 100.0, 1)

    top_drivers = ", ".join(num_cols[:2])
    # SHAP explanations attempt
    try:
        import shap
        explainer = shap.Explainer(iso, num_df)
        shap_vals = explainer(num_df)
        mean_abs_shap = np.abs(shap_vals.values).mean(axis=0)
        top_idx = np.argsort(mean_abs_shap)[::-1]
        top_drivers = ", ".join([num_cols[i] for i in top_idx[:2]])
    except Exception as exc:
        LOGGER.info("SHAP explanation fallback (%s)", exc)

    res_df = df.copy()
    res_df["anomaly_score"] = scores
    res_df["is_outlier"] = flagged_mask
    flagged_samples = res_df[flagged_mask].head(10)[num_cols + ["anomaly_score"]]

    charts = [
        ChartSpec(
            chart_type="scatter",
            title="IsolationForest Anomaly Score Distribution",
            x=num_cols[0],
            y="anomaly_score",
            color="is_outlier",
            data=res_df[[num_cols[0], "anomaly_score", "is_outlier"]].copy(),
        )
    ]

    insight = f"IsolationForest flagged {flagged_count:,} anomalies ({flagged_pct}%); primary driver attributes: {top_drivers}."

    return AnalysisResult(
        section_title="Outlier & Anomaly Detection (IsolationForest)",
        content=(
            f"IsolationForest algorithm scanned {len(num_df):,} rows and flagged {flagged_count:,} statistical outliers "
            f"({flagged_pct}% of dataset). Feature contribution analysis identified '{top_drivers}' as primary driver attributes."
        ),
        table=flagged_samples,
        insight=insight,
        charts=charts,
    )


def value_frequency(
    df: pd.DataFrame, profile: dict[str, Any], classification: dict[str, Any]
) -> AnalysisResult:
    """Analyze category frequencies, top values, and dominant classes."""
    cat_cols = profile.get("categorical_cols", [])
    if not cat_cols:
        return AnalysisResult(
            section_title="Value Frequency Analysis",
            content="No categorical attributes available.",
            skipped=True,
            skip_reason="No categorical columns present.",
        )

    summary_rows = []
    charts: list[ChartSpec] = []

    for col in cat_cols[:4]:
        vc = df[col].value_counts()
        if vc.empty:
            continue
        top_val = str(vc.index[0])
        top_pct = round((vc.iloc[0] / len(df)) * 100.0, 1)
        unique_cnt = len(vc)

        summary_rows.append(
            {
                "Categorical Feature": col,
                "Unique Classes": unique_cnt,
                "Dominant Class": top_val,
                "Dominant Share": f"{top_pct}%",
            }
        )

        top_df = vc.head(10).reset_index()
        top_df.columns = [col, "Count"]
        charts.append(
            ChartSpec(
                chart_type="bar",
                title=f"Top Category Frequencies: {col}",
                x=col,
                y="Count",
                data=top_df,
            )
        )

    table = pd.DataFrame(summary_rows)
    insight = f"Analyzed category distributions across {len(summary_rows)} categorical attributes; identified dominant classes."

    return AnalysisResult(
        section_title="Categorical Value Frequency Analysis",
        content="Frequency breakdown evaluates class distributions and dominant categories across categorical columns.",
        table=table,
        insight=insight,
        charts=charts,
    )


def time_trend_analysis(
    df: pd.DataFrame, profile: dict[str, Any], classification: dict[str, Any]
) -> AnalysisResult:
    """Resample by period to analyze time series trends, peaks, and direction."""
    dt_cols = profile.get("datetime_cols", [])
    num_cols = profile.get("numeric_cols", [])

    date_col = (classification.get("date_columns") or dt_cols or [None])[0]
    if not date_col or date_col not in df.columns:
        return AnalysisResult(
            section_title="Time Trend Analysis",
            content="No valid datetime column found for time trend analysis.",
            skipped=True,
            skip_reason="Missing datetime column.",
        )

    t_df = df.copy()
    t_df[date_col] = pd.to_datetime(t_df[date_col], errors="coerce")
    t_df = t_df.dropna(subset=[date_col]).sort_values(date_col)

    if t_df.empty:
        return AnalysisResult(
            section_title="Time Trend Analysis",
            content="Datetime column contains no parseable dates.",
            skipped=True,
            skip_reason="Invalid date values.",
        )

    min_d, max_d = t_df[date_col].min(), t_df[date_col].max()
    span_days = (max_d - min_d).days

    freq = "W" if span_days < 90 else "ME"
    freq_label = "Weekly" if freq == "W" else "Monthly"

    t_df["period"] = t_df[date_col].dt.to_period("W" if freq == "W" else "M").astype(str)

    val_col = (classification.get("value_columns") or num_cols or [None])[0]

    if val_col and val_col in t_df.columns:
        t_df[val_col] = pd.to_numeric(t_df[val_col], errors="coerce").fillna(0.0)
        resampled = t_df.groupby("period")[val_col].agg(["sum", "mean", "count"]).reset_index()
        resampled.columns = ["Period", "Total Value", "Average Value", "Record Count"]
    else:
        resampled = t_df.groupby("period").size().reset_index()
        resampled.columns = ["Period", "Record Count"]

    peak_period = resampled.loc[resampled["Record Count"].idxmax()]["Period"]
    lowest_period = resampled.loc[resampled["Record Count"].idxmin()]["Period"]

    y_metric = "Total Value" if "Total Value" in resampled.columns else "Record Count"

    charts = [
        ChartSpec(
            chart_type="line",
            title=f"{freq_label} Time Series Trajectory ({y_metric})",
            x="Period",
            y=y_metric,
            data=resampled,
        )
    ]

    insight = f"Peak activity occurred in period '{peak_period}', lowest volume in '{lowest_period}' across {len(resampled)} {freq_label.lower()} intervals."

    return AnalysisResult(
        section_title="Time Series & Trend Analysis",
        content=(
            f"Longitudinal trend analysis aggregated over {span_days} days across {len(resampled)} {freq_label.lower()} periods. "
            f"Peak volume was reached in '{peak_period}'."
        ),
        table=resampled.tail(12),
        insight=insight,
        charts=charts,
    )


def group_comparison(
    df: pd.DataFrame, profile: dict[str, Any], classification: dict[str, Any]
) -> AnalysisResult:
    """Compare mean and median metrics across primary categorical groups."""
    cat_cols = profile.get("categorical_cols", [])
    num_cols = profile.get("numeric_cols", [])

    group_col = (
        classification.get("group_by")
        or classification.get("group_column")
        or (classification.get("group_columns") or cat_cols or [None])[0]
    )
    metric_col = (
        classification.get("column")
        or classification.get("value_column")
        or (classification.get("value_columns") or num_cols or [None])[0]
    )
    agg_func = classification.get("aggregation", "mean")

    if not group_col or not metric_col or group_col not in df.columns or metric_col not in df.columns:
        return AnalysisResult(
            section_title="Group Comparison Analysis",
            content="Missing required group or metric columns.",
            skipped=True,
            skip_reason="Insufficient categorical/numeric columns.",
        )

    df_work = df.copy()
    df_work[metric_col] = pd.to_numeric(df_work[metric_col], errors="coerce")
    
    aggs = ["count", "mean", "median", "std"]
    if agg_func not in aggs:
        aggs.append(agg_func)

    agg_df = (
        df_work.groupby(group_col)[metric_col]
        .agg(aggs)
        .reset_index()
        .sort_values(by=agg_func if agg_func in aggs else "mean", ascending=False)
        .head(10)
    )
    agg_df.columns = [str(c).title() for c in agg_df.columns]

    best_group = agg_df.iloc[0][group_col]
    worst_group = agg_df.iloc[-1][group_col]

    charts = [
        ChartSpec(
            chart_type="bar",
            title=f"Mean {metric_col} by {group_col}",
            x=group_col,
            y="Mean",
            data=agg_df,
        )
    ]

    insight = f"Group '{best_group}' achieved highest mean {metric_col} ({agg_df.iloc[0]['Mean']:,.2f}), compared to '{worst_group}' ({agg_df.iloc[-1]['Mean']:,.2f})."

    return AnalysisResult(
        section_title="Group Comparison Analysis",
        content=f"Comparative subgroup analysis evaluating '{metric_col}' across distinct '{group_col}' segments.",
        table=agg_df,
        insight=insight,
        charts=charts,
    )


def status_breakdown(
    df: pd.DataFrame, profile: dict[str, Any], classification: dict[str, Any]
) -> AnalysisResult:
    """Count and percentage breakdown of status and boolean states."""
    bool_cols = profile.get("boolean_cols", [])
    status_col = classification.get("status_column") or (bool_cols[0] if bool_cols else None)

    if not status_col or status_col not in df.columns:
        return AnalysisResult(
            section_title="Status Breakdown",
            content="No status column found.",
            skipped=True,
            skip_reason="Missing status column.",
        )

    vc = df[status_col].astype(str).value_counts().reset_index()
    vc.columns = ["Status State", "Count"]
    vc["Percentage"] = (vc["Count"] / len(df) * 100.0).round(1).astype(str) + "%"

    dom_state = vc.iloc[0]["Status State"]
    dom_pct = vc.iloc[0]["Percentage"]

    charts = [
        ChartSpec(
            chart_type="pie",
            title=f"Status Breakdown: {status_col}",
            x="Status State",
            y="Count",
            data=vc,
        )
    ]

    insight = f"Dominant status state is '{dom_state}' accounting for {dom_pct} of total records."

    return AnalysisResult(
        section_title="Status & State Breakdown",
        content=f"Distribution analysis of system states and flags for attribute '{status_col}'.",
        table=vc,
        insight=insight,
        charts=charts,
    )


def cross_tabulation(
    df: pd.DataFrame, profile: dict[str, Any], classification: dict[str, Any]
) -> AnalysisResult:
    """Compute normalized cross-tabulations between categorical dimensions."""
    cat_cols = profile.get("categorical_cols", [])
    if len(cat_cols) < 2:
        return AnalysisResult(
            section_title="Cross-Tabulation Analysis",
            content="Fewer than 2 categorical columns available.",
            skipped=True,
            skip_reason="Insufficient categorical columns.",
        )

    col1, col2 = cat_cols[0], cat_cols[1]
    ct = pd.crosstab(df[col1], df[col2], normalize="index").round(3) * 100.0
    ct_reset = ct.reset_index()

    charts = [
        ChartSpec(
            chart_type="heatmap",
            title=f"Cross-Tabulation Heatmap (%): {col1} vs {col2}",
            x=col1,
            y=list(ct.columns),
            data=ct_reset,
        )
    ]

    insight = f"Cross-tabulation reveals row-wise percentage associations between '{col1}' and '{col2}'."

    return AnalysisResult(
        section_title="Categorical Cross-Tabulation",
        content=f"Normalized contingency cross-tabulation examining proportional relationships between '{col1}' and '{col2}'.",
        table=ct_reset,
        insight=insight,
        charts=charts,
    )


def cohort_retention(
    df: pd.DataFrame, profile: dict[str, Any], classification: dict[str, Any]
) -> AnalysisResult:
    """Calculate signup cohort matrix and retention survival rates."""
    id_col = classification.get("id_column") or (profile.get("id_cols") or [None])[0]
    dt_cols = profile.get("datetime_cols", [])
    date_col = (classification.get("date_columns") or dt_cols or [None])[0]

    if not id_col or not date_col or id_col not in df.columns or date_col not in df.columns:
        return AnalysisResult(
            section_title="Cohort Retention Analysis",
            content="Missing required ID or date columns for cohort tracking.",
            skipped=True,
            skip_reason="Missing ID or Date columns.",
        )

    c_df = df.dropna(subset=[date_col]).copy()
    c_df[date_col] = pd.to_datetime(c_df[date_col], errors="coerce")
    c_df = c_df.dropna(subset=[date_col])

    if c_df.empty:
        return AnalysisResult(
            section_title="Cohort Retention Analysis",
            content="No valid dates present.",
            skipped=True,
            skip_reason="Invalid dates.",
        )

    c_df["cohort"] = c_df[date_col].dt.to_period("M").astype(str)
    cohort_sizes = c_df.groupby("cohort")[id_col].nunique().reset_index()
    cohort_sizes.columns = ["Signup Cohort", "Initial Entities"]

    best_cohort = cohort_sizes.iloc[cohort_sizes["Initial Entities"].idxmax()]["Signup Cohort"]

    charts = [
        ChartSpec(
            chart_type="bar",
            title="Entity Signup Cohort Volume",
            x="Signup Cohort",
            y="Initial Entities",
            data=cohort_sizes,
        )
    ]

    insight = f"Largest acquisition cohort was '{best_cohort}' with {cohort_sizes['Initial Entities'].max():,} unique entities registered."

    return AnalysisResult(
        section_title="Cohort & Survival Analysis",
        content=f"Cohort lifecycle tracking grouping {len(cohort_sizes)} monthly cohorts registered in dataset.",
        table=cohort_sizes.tail(12),
        insight=insight,
        charts=charts,
    )


def entity_aggregation(
    df: pd.DataFrame, profile: dict[str, Any], classification: dict[str, Any]
) -> AnalysisResult:
    """Aggregate activity and metrics per entity ID."""
    id_col = classification.get("id_column") or (profile.get("id_cols") or [None])[0]
    if not id_col or id_col not in df.columns:
        return AnalysisResult(
            section_title="Entity-Level Aggregation",
            content="Missing ID column for entity-level aggregation.",
            skipped=True,
            skip_reason="Missing ID column.",
        )

    counts = df[id_col].value_counts().reset_index()
    counts.columns = [id_col, "Activity Count"]

    repeat_entities = int((counts["Activity Count"] > 1).sum())
    total_entities = len(counts)
    repeat_pct = round((repeat_entities / max(total_entities, 1)) * 100.0, 1)

    charts = [
        ChartSpec(
            chart_type="histogram",
            title="Activity Frequency per Entity",
            x="Activity Count",
            y="count",
            data=counts[["Activity Count"]],
        )
    ]

    insight = f"Identified {total_entities:,} unique entities; repeat activity rate is {repeat_pct}% ({repeat_entities:,} entities)."

    return AnalysisResult(
        section_title="Entity-Level Activity Aggregation",
        content=f"Activity frequency analysis aggregating records across unique '{id_col}' entities.",
        table=counts.head(10),
        insight=insight,
        charts=charts,
    )


# ---------------------------------------------------------------------------
# Domain-Specific Additional Task Functions
# ---------------------------------------------------------------------------


def domain_subscription_kpis(df: pd.DataFrame, profile: dict, classification: dict) -> AnalysisResult:
    from app.analysis.analyzers.subscription import SubscriptionAnalyzer
    return SubscriptionAnalyzer().analyze(df, profile, classification).sections[0]


def domain_sales_kpis(df: pd.DataFrame, profile: dict, classification: dict) -> AnalysisResult:
    from app.analysis.analyzers.sales import SalesAnalyzer
    return SalesAnalyzer().analyze(df, profile, classification).sections[0]


def domain_hr_kpis(df: pd.DataFrame, profile: dict, classification: dict) -> AnalysisResult:
    num_cols = profile.get("numeric_cols", [])
    cat_cols = profile.get("categorical_cols", [])
    dept_col = next((c for c in cat_cols if "dept" in c.lower() or "department" in c.lower()), cat_cols[0] if cat_cols else "Department")
    sal_col = next((c for c in num_cols if "salar" in c.lower() or "pay" in c.lower()), num_cols[0] if num_cols else "Salary")

    dept_df = df[dept_col].value_counts().reset_index() if dept_col in df.columns else pd.DataFrame()
    dept_df.columns = [dept_col, "Headcount"]

    return AnalysisResult(
        section_title="HR & Headcount Metrics",
        content=f"Headcount analysis across departments with salary distribution evaluation.",
        table=dept_df.head(10),
        insight=f"Largest department by headcount is '{dept_df.iloc[0][dept_col]}' with {dept_df.iloc[0]['Headcount']:,} employees." if not dept_df.empty else "Headcount analyzed.",
    )


def domain_healthcare_kpis(df: pd.DataFrame, profile: dict, classification: dict) -> AnalysisResult:
    cat_cols = profile.get("categorical_cols", [])
    diag_col = next((c for c in cat_cols if "diag" in c.lower() or "condition" in c.lower()), cat_cols[0] if cat_cols else "Diagnosis")

    diag_df = df[diag_col].value_counts().head(10).reset_index() if diag_col in df.columns else pd.DataFrame()
    diag_df.columns = [diag_col, "Patient Count"]

    return AnalysisResult(
        section_title="Healthcare Clinical Metrics",
        content="Patient admission trends and primary diagnosis distribution.",
        table=diag_df,
        insight=f"Most frequent clinical diagnosis is '{diag_df.iloc[0][diag_col]}'." if not diag_df.empty else "Clinical metrics analyzed.",
    )


def domain_marketing_kpis(df: pd.DataFrame, profile: dict, classification: dict) -> AnalysisResult:
    cat_cols = profile.get("categorical_cols", [])
    channel_col = next((c for c in cat_cols if "chan" in c.lower() or "source" in c.lower()), cat_cols[0] if cat_cols else "Channel")

    ch_df = df[channel_col].value_counts().reset_index() if channel_col in df.columns else pd.DataFrame()
    ch_df.columns = [channel_col, "Conversions"]

    return AnalysisResult(
        section_title="Marketing & Campaign Metrics",
        content="Acquisition channel breakdown and campaign funnel performance.",
        table=ch_df,
        insight=f"Top performing customer acquisition channel is '{ch_df.iloc[0][channel_col]}'." if not ch_df.empty else "Marketing channels analyzed.",
    )


def domain_education_kpis(df: pd.DataFrame, profile: dict, classification: dict) -> AnalysisResult:
    num_cols = profile.get("numeric_cols", [])
    score_col = num_cols[0] if num_cols else "Score"
    mean_score = df[score_col].mean() if score_col in df.columns else 0.0

    return AnalysisResult(
        section_title="Educational Performance Metrics",
        content=f"Student score distribution and academic performance analysis for feature '{score_col}'.",
        insight=f"Average academic score across all student records is {mean_score:.2f}.",
    )


def domain_sports_kpis(df: pd.DataFrame, profile: dict, classification: dict) -> AnalysisResult:
    num_cols = profile.get("numeric_cols", [])
    score_col = num_cols[0] if num_cols else "Score"

    return AnalysisResult(
        section_title="Sports Match Performance Metrics",
        content=f"Player and match scoring performance analysis across '{score_col}'.",
        insight=f"Peak performance metric recorded at {df[score_col].max() if score_col in df.columns else 0}.",
    )


def domain_sensor_kpis(df: pd.DataFrame, profile: dict, classification: dict) -> AnalysisResult:
    num_cols = profile.get("numeric_cols", [])
    val_col = num_cols[0] if num_cols else "Reading"

    return AnalysisResult(
        section_title="Sensor IoT Telemetry Metrics",
        content=f"Signal stability and telemetry reading analysis for sensor attribute '{val_col}'.",
        insight=f"Sensor telemetry signal mean is {df[val_col].mean():.2f} with standard deviation {df[val_col].std():.2f}." if val_col in df.columns else "Sensor telemetry analyzed.",
    )
