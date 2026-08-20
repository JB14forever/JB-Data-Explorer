# ==================================================================================
#  FILE: agents/nlp_agent.py
# ==================================================================================
#  WHAT THIS FILE DOES (in plain English):
#  This is the agent behind the "Ask a question in plain English" feature
#  (Tab 4 of the app). The user types a question like "Show me churn by
#  contract type", and this agent asks the AI to work out three things:
#    1. What data transformation is needed to answer it (as pandas code)
#    2. Which chart type best communicates the answer
#    3. A written description of the chart and a data-driven narrative
#
#  IMPORTANT: the AI is only ever asked to describe the data and choose
#  chart settings — it is given column statistics that were already
#  calculated by ordinary pandas code, and it is never asked to invent a
#  number itself. This separation is the core safeguard discussed in the
#  Client Report, Section 2.4 ("Data Processing Strategy").
# ==================================================================================

import json
import datetime
import pandas as pd
import numpy as np
import streamlit as st
from utils.llm_client import get_llm_client, LLM_MODEL


class NLPAgent:
    """
    Intelligent Natural Language → Graph engine.
    Translates user questions into rich chart specifications with
    LLM-chosen chart types, proper labels, figure descriptions,
    and dataset-driven contextual narratives.
    """

    def __init__(self):
        self.client = get_llm_client()
        self.available = self.client is not None

    def query(self, question: str, df: pd.DataFrame, columns: list, domain_context: dict = None) -> dict:
        """
        Translates a natural language question into a full chart specification.

        High-level flow:
          1. Summarise every column's type and basic statistics (min/max/
             mean for numbers, top values for categories) — NOT the raw
             data itself, keeping what's sent to the AI small and private.
          2. Send that summary plus the user's question to the AI, with a
             very detailed instruction set (the `system_prompt` below)
             describing exactly which chart types are supported and how
             to choose between them.
          3. Parse the AI's JSON answer into a chart specification that
             app.py can use to actually draw the chart with Plotly.
        """
        if not self.available:
            return {
                "filter_code": None,
                "chart_type": None,
                "chart_config": {},
                "figure_description": None,
                "data_narrative": None,
                "error": "LLM API not configured. Natural language queries disabled."
            }

        try:
            # ── Build a compact statistical summary of every column ──
            # This is what gives the AI enough context to answer sensibly
            # without ever seeing the full, potentially sensitive dataset.
            dtypes_info = {}
            stats_info = {}
            for col in columns:
                if col not in df.columns:
                    continue
                dtypes_info[col] = str(df[col].dtype)
                try:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        stats_info[col] = {
                            "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                            "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                            "mean": round(float(df[col].mean()), 2) if not pd.isna(df[col].mean()) else None,
                            "unique_count": int(df[col].nunique())
                        }
                    else:
                        top_vals = df[col].value_counts().head(8).to_dict()
                        stats_info[col] = {
                            "unique_count": int(df[col].nunique()),
                            "top_values": {str(k): int(v) for k, v in top_vals.items()}
                        }
                except Exception:
                    stats_info[col] = {"unique_count": int(df[col].nunique())}

            domain_str = ""
            if domain_context:
                domain_str = f"""
                Domain Context:
                - Industry: {domain_context.get('industry', 'Unknown')}
                - Target Variable: {domain_context.get('target_variable', 'Unknown')}
                - Business Summary: {domain_context.get('business_summary', 'N/A')}
                """

            # ── The full instruction set given to the AI ──
            # This is deliberately very detailed: it lists every supported
            # chart type, gives explicit rules for choosing 2D vs 3D, and
            # locks the AI into returning ONLY a specific JSON shape. This
            # is one of the "guardrails" mentioned in the Research Paper,
            # Section 4.2 — it stops the AI from inventing unsupported
            # chart types or returning free-form, hard-to-parse text.
            system_prompt = f"""You are an expert data visualization analyst. The user has a pandas DataFrame `df` with:

COLUMNS AND DTYPES:
{json.dumps(dtypes_info, indent=2)}

COLUMN STATISTICS:
{json.dumps(stats_info, indent=2, default=str)}

{domain_str}

TOTAL ROWS: {len(df)}

You are an expert data visualization analyst. The user will ask a question in natural language about a dataset.

Your job is to:
1. Deeply understand the question intent (comparison, distribution, trend, relationship, composition, etc.)
2. Inspect the data shape, cardinality, and variable types to determine the best visualization strategy
3. Decide whether a 2D or 3D chart is more appropriate (see decision rules below)
4. Select the single most suitable chart type from the supported list
5. Write valid pandas code to prepare/aggregate the data
6. Define professional chart metadata (title, axis labels, legend)
7. Write a figure caption suitable for an analytical report
8. Write a data-driven narrative about patterns, distributions, and insights from the actual values

────────────────────────────────────────
2D vs 3D DECISION RULES
────────────────────────────────────────

Use a 3D chart ONLY when ALL of the following are true:
  - There are exactly THREE continuous/numeric dimensions that all carry meaningful information simultaneously
  - The 3D perspective genuinely reveals a pattern (e.g., a surface, a cluster volume, a trajectory)
    that a 2D chart with color/size encoding would obscure or flatten
  - The audience is analytical and expects exploratory depth (not a summary/executive view)
  - The number of data points is sufficient to justify the added complexity (typically >50 points)

Use a 2D chart in ALL other cases, including:
  - When one of the three variables can be encoded as color, size, or facet instead
  - When the goal is comparison, ranking, or composition (bar, pie, treemap, etc.)
  - When the dataset is small or categorical
  - When the output is for a report, dashboard, or non-technical audience
  - When clarity and immediate readability matter more than depth exploration

────────────────────────────────────────
SUPPORTED CHART TYPES
────────────────────────────────────────

2D CHARTS (default — use unless 3D rules are met):
  "bar"        → Categorical comparisons, rankings, counts
  "histogram"  → Single-variable distribution, frequency analysis
  "line"       → Trends over time or ordered sequences
  "scatter"    → Relationship / correlation between two numeric variables
  "box"        → Distribution spread, median, outliers across groups
  "pie"        → Proportional composition (ONLY when ≤ 8 categories)
  "violin"     → Distribution shape comparison across groups
  "area"       → Cumulative or stacked trends over time
  "treemap"    → Hierarchical proportions and part-to-whole
  "funnel"     → Sequential stage drop-off or conversion rates
  "sunburst"   → Nested categorical hierarchies (2+ levels)
  "heatmap"    → Matrix correlations, pivot tables, or cross-tabulations

3D CHARTS (use only when 3D decision rules above are satisfied):
  "scatter_3d" → 3-variable numeric relationship / cluster exploration
  "surface_3d" → Continuous surface over a 2D grid (e.g., Z = f(X, Y))
  "line_3d"    → 3D trajectory or path through numeric space
  "bar_3d"     → Grouped bar across two categorical axes with numeric height

────────────────────────────────────────
AXIS AND FORMATTING RULES
────────────────────────────────────────
- For 'bar', 'line', and 'scatter' charts, ALWAYS specify BOTH 'x' and 'y' in chart_config.
- If you want to show the frequency/count of a single variable, use 'histogram' (do NOT use 'bar' with a null y).
- Avoid returning wide-form DataFrames with mixed data types. If comparing multiple metrics, `melt` the DataFrame into long-form first using pandas.

────────────────────────────────────────
STEP-BY-STEP REASONING (internal — do not include in output)
────────────────────────────────────────

Before producing the JSON, reason through these questions silently:
  a) What is the user's analytical intent? (compare / distribute / correlate / compose / trend / explore)
  b) How many variables are involved? What are their types (categorical, continuous, temporal)?
  c) Does a 3rd numeric dimension exist that CANNOT be adequately encoded as color or size?
  d) Would a 3D chart genuinely add insight, or would it introduce unnecessary visual complexity?
  e) Which specific chart type best serves the intent with the least complexity?
  f) What pandas transformation is needed (groupby, pivot, resample, melt, etc.)?

────────────────────────────────────────
OUTPUT FORMAT
────────────────────────────────────────

Respond ONLY with a valid JSON object. No markdown, no backticks, no explanation outside the JSON.

{{
    "dimension": "2D or 3D",
    "dimension_rationale": "One concise sentence explaining why 2D or 3D was chosen for this specific question and data.",
    "filter_code": "A single valid pandas expression using `df` that produces the chart-ready DataFrame. Use .reset_index() after any aggregation. Return null only if df can be used directly without transformation.",
    "chart_type": "Exactly one chart type from the supported list above (e.g., 'bar', 'scatter_3d')",
    "chart_config": {{
        "title": "A descriptive, professional chart title that reflects the actual question and data",
        "x": "column name for the x-axis",
        "y": "column name for the y-axis (use null for single-axis charts like histogram or pie)",
        "z": "column name for the z-axis if 3D, otherwise null",
        "color": "column name for color grouping/encoding, or null if not applicable",
        "size": "column name for bubble/marker size encoding, or null if not applicable",
        "labels": {{"original_column_name": "Human-Readable Axis Label"}},
        "legend_title": "A short, clear legend title, or null if no legend"
    }},
    "figure_description": "A 2–3 sentence professional figure caption. Describe what the chart visualizes, the variables shown, and the scope of the data. Suitable for inclusion in an analytical report or presentation.",
    "data_narrative": "A 3–5 sentence insight narrative grounded in actual data values. Highlight the most important patterns, comparisons, distributions, outliers, or trends visible in the data. Cite specific values, percentages, or group names where possible. Do NOT explain chart type choices."
}}"""

            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.1
            )
            raw_content = response.choices[0].message.content.strip()

            # Clean up markdown formatting if present (the AI sometimes
            # wraps its JSON answer in code fences despite being asked not to)
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            if raw_content.startswith("```"):
                raw_content = raw_content[3:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]
            raw_content = raw_content.strip()

            parsed = json.loads(raw_content)

            return {
                "filter_code": parsed.get("filter_code"),
                "chart_type": parsed.get("chart_type"),
                "chart_config": parsed.get("chart_config", {}),
                "figure_description": parsed.get("figure_description"),
                "data_narrative": parsed.get("data_narrative"),
            }

        except Exception as e:
            # Any failure (AI unreachable, malformed JSON, etc.) is
            # reported back as a friendly error rather than crashing the app.
            return {
                "filter_code": None,
                "chart_type": None,
                "chart_config": {},
                "figure_description": None,
                "data_narrative": None,
                "error": f"Failed to process query: {str(e)}"
            }

    def log_query(self, question: str, response: dict):
        """
        Logs the user prompt and agent response to Streamlit session state.
        This keeps a running history of every question asked during the
        current session, purely for reference — it is not currently
        displayed in the UI but is available for future auditing features.
        """
        if 'query_log' not in st.session_state:
            st.session_state['query_log'] = []

        st.session_state['query_log'].append({
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'question': question,
            'response': response
        })
