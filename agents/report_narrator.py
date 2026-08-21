# ==================================================================================
#  FILE: agents/report_narrator.py
# ==================================================================================
#  Writes the prose sections of the final PDF report: title, executive
#  summary, cleaning explanation, model interpretation, and conclusions.
#  Last AI agent in the pipeline, and it always works from results that
#  were already calculated by the other agents (cleaning logs, ML
#  leaderboard, etc). It turns numbers and logs into readable business
#  language, it doesn't calculate anything new.
# ==================================================================================

"""
Report Narrator Agent, Generates LLM-powered narrative sections for the PDF report.
"""

import json
from utils.llm_client import get_llm_client, LLM_MODEL


class ReportNarrator:
    def __init__(self):
        self.client = get_llm_client()
        self.available = self.client is not None

    def _call_llm(self, system: str, user: str) -> str:
        """Shared helper the methods below use to send a prompt and get
        plain text back. If the AI isn't available or the call fails, an
        empty string comes back instead of crashing, the PDF generator
        already handles an empty narrative by showing a placeholder line."""
        if not self.available:
            return ""
        try:
            r = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.3
            )
            return r.choices[0].message.content.strip()
        except Exception:
            return ""

    # ── report title ──────────────────────────────────────────────
    def generate_report_title(self, domain_context: dict, dataset_name: str) -> str:
        """Generates a short title for the report cover page."""
        prompt = f"""Generate a professional, concise analytical report title (max 10 words) for a dataset named '{dataset_name}'.
Industry: {domain_context.get('industry','General')}. Summary: {domain_context.get('business_summary','Data analysis report.')}.
Return ONLY the title text, no quotes or formatting."""
        return self._call_llm(prompt, "Generate title.") or f"{dataset_name}: Analytical Report"

    # ── executive summary (opening page of the report) ──────────────
    def generate_executive_summary(self, domain_context: dict, ml_results: dict, cleaning_logs, dataset_name: str) -> str:
        """Writes the opening executive summary, grounded in the actual
        industry context, cleaning results, and best model score already
        computed earlier in the pipeline."""
        ml_str = ""
        if ml_results and ml_results.get('best_model_name'):
            ml_str = f"Best model: {ml_results['best_model_name']} ({ml_results.get('metric_name','')}: {ml_results.get('best_metric_value',0):.4f}). Task: {ml_results.get('task_type','')}."

        clean_count = len(cleaning_logs) if isinstance(cleaning_logs, list) else len(cleaning_logs) if isinstance(cleaning_logs, dict) else 0

        prompt = f"""Write a highly detailed executive summary for a data analytics report. It MUST be EXACTLY 12 sentences long, perfectly formatted into exactly 3 paragraphs containing exactly 4 sentences each.
Dataset: {dataset_name}
Industry: {domain_context.get('industry','General')}
Business: {domain_context.get('business_summary','N/A')}
Target: {domain_context.get('target_variable','N/A')}
Cleaning: {clean_count} columns required intervention.
{ml_str}
Include specific figures, important facts, and insights gained during the cleaning and ML sweeping stages about the dataset. Be highly descriptive. Professional tone. No headers. Return ONLY the summary text."""
        return self._call_llm(prompt, "Generate executive summary.")

    # ── cleaning narrative (section 2 of the PDF) ────────────────────
    def generate_cleaning_narrative(self, cleaning_logs) -> str:
        """Turns the raw cleaning decision log into a paragraph
        explaining what got done and why."""
        logs_str = json.dumps(cleaning_logs[:10] if isinstance(cleaning_logs, list) else cleaning_logs, default=str)
        prompt = f"""Write a highly detailed, 8-10 sentence narrative elaborating on the data cleaning decisions.
CLEANING LOG: {logs_str}
Mention key actions taken and provide descriptive justifications for why they were necessary. Be highly descriptive. Professional tone. No headers. Return ONLY text."""
        return self._call_llm(prompt, "Summarize cleaning.")

    # ── model interpretation (section 4 of the PDF) ───────────────────
    def generate_ml_interpretation(self, ml_results: dict, domain_context: dict) -> str:
        """Explains the leaderboard and feature importance in plain
        language. Returns an empty string if nothing's been trained yet,
        so the section can just get skipped."""
        if not ml_results or not ml_results.get('leaderboard'):
            return ""
        lb = json.dumps(ml_results.get('leaderboard', [])[:5], default=str)
        fi = json.dumps(dict(list(ml_results.get('feature_importance', {}).items())[:5]), default=str)
        prompt = f"""Write a highly descriptive interpretation of ML model results. It MUST be exactly 1 paragraph and no more than 2 sentences maximum.
Task: {ml_results.get('task_type','')}. Best: {ml_results.get('best_model_name','')}.
Metric: {ml_results.get('metric_name','')}: {ml_results.get('best_metric_value',0):.4f}.
Leaderboard: {lb}
Top Features: {fi}
Industry: {domain_context.get('industry','General')}, Target: {domain_context.get('target_variable','N/A')}.
Elaborate deeply on performance, compare the models, and thoroughly discuss why the top features are important. Be very descriptive. Professional tone. No headers. Do not exceed 3 sentences."""
        return self._call_llm(prompt, "Interpret ML results.")

    # ── conclusions and recommendations (closing section) ──────────────
    def generate_conclusions(self, domain_context: dict, ml_results: dict, saved_queries: list) -> str:
        """Writes the closing conclusions, drawing together the domain
        context, best model result, and whatever natural-language
        questions the user chose to save into the report."""
        queries = [q.get('question', '') for q in (saved_queries or [])[:5]]
        ml_str = ""
        if ml_results and ml_results.get('best_model_name'):
            ml_str = f"Best model: {ml_results['best_model_name']} ({ml_results.get('metric_name','')}: {ml_results.get('best_metric_value',0):.4f})."
        prompt = f"""Write a comprehensive, highly detailed 10-12 sentence conclusion and recommendation section for a data analytics report.
Industry: {domain_context.get('industry','General')}. Target: {domain_context.get('target_variable','N/A')}.
Business: {domain_context.get('business_summary','N/A')}. {ml_str}
Analyst queries explored: {queries}
Include: key takeaways, elaborate actionable recommendations, and suggested next steps. Be very descriptive.
Professional tone. No headers. Return ONLY text."""
        return self._call_llm(prompt, "Generate conclusions.")
