# ==================================================================================
#  FILE: agents/graph_describer.py
# ==================================================================================
#  Whenever the app draws a distribution chart or the correlation heatmap
#  in the EDA tab, this agent writes a short plain-English paragraph
#  explaining what it shows. Same rule as the other agents: it only ever
#  gets numbers that were already calculated by pandas/scipy, it explains
#  them, it doesn't calculate anything new.
# ==================================================================================

"""
Graph Description Agent, Generates contextual chart descriptions using LLM.
"""

import json
import pandas as pd
import numpy as np
from utils.llm_client import get_llm_client, LLM_MODEL


class GraphDescriber:
    def __init__(self):
        self.client = get_llm_client()
        self.available = self.client is not None

    # ── describe a single column's distribution chart ────────────
    def describe_distribution(self, df: pd.DataFrame, column: str, domain_context: dict = None) -> str:
        """Writes a short paragraph about a column's histogram, using
        real, already-calculated stats (mean, median, spread, skewness,
        outlier boundaries) as the only source of numbers. Falls back to
        _fallback_description if the AI isn't available."""
        if not self.available:
            return self._fallback_description(df, column)

        col_data = df[column]
        stats = {}
        if pd.api.types.is_numeric_dtype(col_data):
            stats = {
                "dtype": "numeric", "count": int(col_data.count()),
                "mean": round(float(col_data.mean()), 4) if not pd.isna(col_data.mean()) else None,
                "median": round(float(col_data.median()), 4) if not pd.isna(col_data.median()) else None,
                "std": round(float(col_data.std()), 4) if not pd.isna(col_data.std()) else None,
                "min": round(float(col_data.min()), 4), "max": round(float(col_data.max()), 4),
                "skewness": round(float(col_data.skew()), 4) if not pd.isna(col_data.skew()) else None,
                "null_count": int(col_data.isnull().sum()), "unique_values": int(col_data.nunique()),
                "q25": round(float(col_data.quantile(0.25)), 4), "q75": round(float(col_data.quantile(0.75)), 4),
            }
        else:
            vc = col_data.value_counts().head(15)
            stats = {
                "dtype": "categorical", "count": int(col_data.count()),
                "unique_values": int(col_data.nunique()), "null_count": int(col_data.isnull().sum()),
                "top_values": {str(k): int(v) for k, v in vc.items()},
                "mode": str(col_data.mode().iloc[0]) if not col_data.mode().empty else "N/A",
            }

        domain_str = ""
        if domain_context:
            domain_str = f"Industry: {domain_context.get('industry','Unknown')}. Target: {domain_context.get('target_variable','Unknown')}. Summary: {domain_context.get('business_summary','N/A')}."

        prompt = f"""You are an expert data analyst. Describe a histogram+boxplot chart for column "{column}".
STATS: {json.dumps(stats, default=str)}
{domain_str}
Cover: axes meaning, distribution shape, central tendency, spread, outliers, business interpretation.
4-6 sentences, professional tone. No headers. Return ONLY the description text."""

        try:
            r = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": f"Describe: {column}"}],
                temperature=0.2
            )
            return r.choices[0].message.content.strip()
        except Exception:
            return self._fallback_description(df, column)

    # ── describe the correlation heatmap ──────────────────────────
    def describe_heatmap(self, df: pd.DataFrame, domain_context: dict = None) -> str:
        """Writes a short paragraph about which pairs of numeric columns
        are most strongly related on the heatmap. The correlation values
        (corr()) come from pandas, the AI only gets the resulting numbers
        and explains what they mean in business terms."""
        if not self.available:
            return "Correlation heatmap showing pairwise Pearson correlations between numeric features."

        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty or numeric_df.shape[1] < 2:
            return "Insufficient numeric columns for correlation analysis."

        corr = numeric_df.corr().round(3)
        strong = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) > 0.3:
                    strong.append({"col1": corr.columns[i], "col2": corr.columns[j], "r": round(val, 3)})
        strong.sort(key=lambda x: abs(x["r"]), reverse=True)

        domain_str = ""
        if domain_context:
            domain_str = f"Industry: {domain_context.get('industry','Unknown')}. Target: {domain_context.get('target_variable','Unknown')}."

        prompt = f"""Describe a Pearson correlation heatmap with {len(corr.columns)} numeric features.
STRONG CORRELATIONS (|r|>0.3): {json.dumps(strong[:10])}
FEATURES: {list(corr.columns)}
{domain_str}
Cover: what heatmap shows, strongest correlations, feature clusters, target relevance, multicollinearity.
4-6 sentences, professional tone. No headers. Return ONLY description text."""

        try:
            r = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": "Describe heatmap."}],
                temperature=0.2
            )
            return r.choices[0].message.content.strip()
        except Exception:
            return "Correlation heatmap showing pairwise Pearson correlations between numeric features."

    # ── simple, non-AI backup description ─────────────────────────
    def _fallback_description(self, df: pd.DataFrame, column: str) -> str:
        """Only used when the AI is unavailable, builds a short
        template-based sentence straight from the stats, so the app still
        shows something useful instead of an empty box."""
        col_data = df[column]
        if pd.api.types.is_numeric_dtype(col_data):
            return (f"Distribution of '{column}': mean={col_data.mean():.2f}, median={col_data.median():.2f}, "
                    f"std={col_data.std():.2f}, range=[{col_data.min():.2f}, {col_data.max():.2f}], n={col_data.count()}.")
        else:
            return (f"Distribution of '{column}': {col_data.nunique()} unique values, n={col_data.count()}. "
                    f"Most frequent: '{col_data.mode().iloc[0]}' ({col_data.value_counts().iloc[0] / len(col_data) * 100:.1f}%).")
