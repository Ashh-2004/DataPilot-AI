"""Self-contained HTML report generator for DataPilot AI dataset reports."""

import html
from typing import Any


def render_report_html(report: dict[str, Any]) -> str:
    """Generate a self-contained, beautifully styled HTML dataset report with inline CSS (no CDN dependencies)."""
    table_name = html.escape(str(report.get("table_name", "Dataset")))
    version = report.get("version", 1)
    rows_raw = report.get("rows_raw", 0)
    rows_clean = report.get("rows_clean", 0)

    quality = report.get("quality", {})
    raw_score = quality.get("raw_score", 0)
    clean_score = quality.get("clean_score", 0)
    issues = quality.get("issues", [])

    domain_info = report.get("domain", {})
    domain = html.escape(str(domain_info.get("domain", "General Data")))
    confidence = domain_info.get("confidence", 1.0)
    reason = html.escape(str(domain_info.get("reason", "")))
    kpis = [html.escape(str(k)) for k in domain_info.get("likely_kpis", [])]
    dims = [html.escape(str(d)) for d in domain_info.get("likely_dimensions", [])]
    questions = [html.escape(str(q)) for q in domain_info.get("suggested_questions", [])]

    profile = report.get("profile", {})
    columns = profile.get("columns", [])
    correlations = profile.get("correlations", [])

    # HTML Building Blocks
    issues_html = ""
    if issues:
        for issue in issues:
            sev = str(issue.get("severity", "medium")).lower()
            badge_class = f"badge-{sev}"
            col_str = html.escape(str(issue.get("column"))) if issue.get("column") else "Table-level"
            desc = html.escape(str(issue.get("description", "")))
            sug = html.escape(str(issue.get("suggestion", "")))

            issues_html += f"""
            <div class="card issue-card">
                <div class="issue-header">
                    <span class="badge {badge_class}">{sev.upper()}</span>
                    <strong>Target: {col_str}</strong>
                </div>
                <p class="issue-desc">{desc}</p>
                <div class="suggestion">💡 <strong>Suggestion:</strong> {sug}</div>
            </div>
            """
    else:
        issues_html = '<div class="card"><p style="color:#4caf50;">✅ No major data quality issues detected!</p></div>'

    columns_rows_html = ""
    for col in columns:
        c_name = html.escape(str(col.get("name")))
        c_type = html.escape(str(col.get("type")))
        null_pct = col.get("null_pct", 0.0)
        dist_cnt = col.get("distinct_count", 0)

        stats_parts = []
        if col.get("min") is not None and col.get("max") is not None:
            stats_parts.append(f"Range: [{col['min']} .. {col['max']}]")
        if col.get("mean") is not None:
            stats_parts.append(f"Mean: {col['mean']}")
        if col.get("std") is not None:
            stats_parts.append(f"Std: {col['std']}")
        if col.get("top_5"):
            top_vals = ", ".join([f"{v.get('value')} ({v.get('count')})" for v in col["top_5"][:3]])
            stats_parts.append(f"Top: {html.escape(top_vals)}")
        stats_str = html.escape(" | ".join(stats_parts)) if stats_parts else "N/A"

        flags = []
        if col.get("is_candidate_id"):
            flags.append('<span class="badge badge-info">ID</span>')
        if col.get("is_constant"):
            flags.append('<span class="badge badge-warn">Constant</span>')
        flags_str = " ".join(flags) if flags else "-"

        columns_rows_html += f"""
        <tr>
            <td><code>{c_name}</code></td>
            <td><code>{c_type}</code></td>
            <td>{null_pct:.1f}%</td>
            <td>{dist_cnt:,}</td>
            <td>{stats_str}</td>
            <td>{flags_str}</td>
        </tr>
        """

    questions_html = ""
    for q in questions:
        questions_html += f'<li><span class="question-pill">{q}</span></li>'

    correlations_rows_html = ""
    if correlations:
        for corr in correlations:
            c1 = html.escape(str(corr.get("col1")))
            c2 = html.escape(str(corr.get("col2")))
            c_val = corr.get("correlation", 0.0)
            color = "#4caf50" if abs(c_val) > 0.7 else ("#ff9800" if abs(c_val) > 0.4 else "#9e9e9e")
            correlations_rows_html += f"""
            <tr>
                <td><code>{c1}</code> & <code>{c2}</code></td>
                <td><strong style="color:{color};">{c_val:+.4f}</strong></td>
            </tr>
            """
    else:
        correlations_rows_html = '<tr><td colspan="2">No numeric correlations available.</td></tr>'

    kpis_str = ", ".join(kpis) if kpis else "N/A"
    dims_str = ", ".join(dims) if dims else "N/A"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataPilot AI Report - {table_name}</title>
    <style>
        :root {{
            --bg-primary: #0e1117;
            --bg-secondary: #1e222d;
            --bg-card: #262a36;
            --border-color: #2a2f3d;
            --text-primary: #e0e6ed;
            --text-secondary: #9a9ea8;
            --accent-color: #4f46e5;
            --success-color: #10b981;
            --warn-color: #f59e0b;
            --danger-color: #ef4444;
            --info-color: #3b82f6;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            margin: 0;
            padding: 32px 16px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #1e1b4b 0%, #1e222d 100%);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px 32px;
            margin-bottom: 24px;
        }}
        h1 {{
            margin: 0 0 8px 0;
            font-size: 28px;
            color: #ffffff;
        }}
        .subtitle {{
            color: var(--text-secondary);
            font-size: 14px;
            margin: 0;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .metric-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: 700;
            color: #ffffff;
            margin-top: 4px;
        }}
        .metric-label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
        }}
        .score-bar-bg {{
            background: #374151;
            border-radius: 6px;
            height: 10px;
            width: 100%;
            margin-top: 8px;
            overflow: hidden;
        }}
        .score-bar-fill {{
            height: 100%;
            border-radius: 6px;
        }}
        .section {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        h2 {{
            font-size: 20px;
            margin-top: 0;
            margin-bottom: 16px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
            color: #ffffff;
        }}
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        }}
        .issue-card {{
            border-left: 4px solid var(--warn-color);
        }}
        .issue-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 8px;
        }}
        .issue-desc {{
            margin: 0 0 8px 0;
            font-size: 14px;
        }}
        .suggestion {{
            font-size: 13px;
            color: #a7f3d0;
            background: rgba(16, 185, 129, 0.1);
            padding: 8px 12px;
            border-radius: 6px;
        }}
        .badge {{
            font-size: 11px;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 4px;
            text-transform: uppercase;
        }}
        .badge-high {{ background: var(--danger-color); color: #ffffff; }}
        .badge-medium {{ background: var(--warn-color); color: #000000; }}
        .badge-low {{ background: var(--info-color); color: #ffffff; }}
        .badge-info {{ background: var(--info-color); color: #ffffff; }}
        .badge-warn {{ background: var(--warn-color); color: #000000; }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            text-align: left;
        }}
        th, td {{
            padding: 12px 14px;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            background-color: var(--bg-card);
            color: var(--text-secondary);
            font-weight: 600;
        }}
        code {{
            background: #111827;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
            color: #93c5fd;
        }}
        ul.question-list {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        ul.question-list li {{
            margin-bottom: 8px;
        }}
        .question-pill {{
            display: inline-block;
            background: #312e81;
            color: #e0e7ff;
            padding: 8px 14px;
            border-radius: 20px;
            font-size: 14px;
            border: 1px solid #4338ca;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Dataset Report: <code>{table_name}</code></h1>
            <p class="subtitle">Version {version} | Domain: <strong>{domain}</strong> ({confidence:.0%} confidence) | Generated by DataPilot AI</p>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Raw Quality Score</div>
                <div class="metric-value" style="color: {'#10b981' if raw_score >= 80 else ('#f59e0b' if raw_score >= 50 else '#ef4444')};">{raw_score}/100</div>
                <div class="score-bar-bg">
                    <div class="score-bar-fill" style="width: {raw_score}%; background: {'#10b981' if raw_score >= 80 else ('#f59e0b' if raw_score >= 50 else '#ef4444')};"></div>
                </div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Cleaned Quality Score</div>
                <div class="metric-value" style="color: {'#10b981' if clean_score >= 80 else ('#f59e0b' if clean_score >= 50 else '#ef4444')};">{clean_score}/100</div>
                <div class="score-bar-bg">
                    <div class="score-bar-fill" style="width: {clean_score}%; background: {'#10b981' if clean_score >= 80 else ('#f59e0b' if clean_score >= 50 else '#ef4444')};"></div>
                </div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Total Rows</div>
                <div class="metric-value">{rows_clean:,}</div>
                <div style="font-size:12px; color:var(--text-secondary); margin-top:4px;">Raw: {rows_raw:,} rows</div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Total Columns</div>
                <div class="metric-value">{len(columns)}</div>
                <div style="font-size:12px; color:var(--text-secondary); margin-top:4px;">Domain: {domain}</div>
            </div>
        </div>

        <div class="section">
            <h2>🏷️ Domain Classification & Insights</h2>
            <p><strong>Reasoning:</strong> {reason}</p>
            <p><strong>Likely KPIs:</strong> {kpis_str}</p>
            <p><strong>Likely Dimensions:</strong> {dims_str}</p>

            <h3 style="font-size:16px; margin-top:20px; color:#ffffff;">Suggested Analytical Questions:</h3>
            <ul class="question-list">
                {questions_html}
            </ul>
        </div>

        <div class="section">
            <h2>⚠️ Quality Issues & Remediation List</h2>
            {issues_html}
        </div>

        <div class="section">
            <h2>📋 Column Profiling & Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Type</th>
                        <th>Null %</th>
                        <th>Distinct</th>
                        <th>Stats / Sample</th>
                        <th>Flags</th>
                    </tr>
                </thead>
                <tbody>
                    {columns_rows_html}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>📈 Top Numeric Correlations</h2>
            <table>
                <thead>
                    <tr>
                        <th>Column Pair</th>
                        <th>Pearson Correlation</th>
                    </tr>
                </thead>
                <tbody>
                    {correlations_rows_html}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
