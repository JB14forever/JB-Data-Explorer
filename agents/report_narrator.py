# ==================================================================================
#  FILE: agents/report_narrator.py
# ==================================================================================
#  WHAT THIS FILE DOES (in plain English):
#  This agent writes the narrative (prose) sections of the final downloadable
#  PDF report — the title, executive summary, cleaning explanation, model
#  interpretation, and conclusions. It is the last AI agent in the pipeline
#  and always works from results that have already been calculated by the
#  other agents (cleaning logs, ML leaderboard, etc.) — it turns numbers
#  and logs into readable business language, it does not calculate anything
#  new itself.
# ==================================================================================

"""
Report Narrator Agent — Generates LLM-powered narrative sections for the PDF report.
"""

import json
from utils.llm_client import get_llm_client, LLM_MODEL


class ReportNarrator:
    def __init__(self):
        self.client = get_llm_client()
        self.available = self.client is not None

    def _call_llm(self, system: str, user: str) -> str:
        """Shared helper used by every method below to send a prompt to
        the AI and return plain text. If the AI is unavailable or the call
        fails for any reason, an empty string is returned instead of
        crashing — the PDF generator (utils/pdf_generator.py) already
        handles an empty narrative gracefully by showing a placeholder
        sentence instead."""
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

    # ── REPORT TITLE ──────────────────────────────────────────────────
    def generate_report_title(self, domain_context: dict, dataset_name: str) -> str:
        """Generates a short, professional title for the report cover page."""
        prompt = f"""Generate a professional, concise analytical report title (max 10 words) for a dataset named '{dataset_name}'.
Industry: {domain_context.get('industry','General')}. Summary: {domain_context.get('business_summary','Data analysis report.')}.
Return ONLY the title text, no quotes or formatting."""
        return self._call_llm(prompt, "Generate title.") or f"{dataset_name} — Analytical Report"

    # ── EXECUTIVE SUMMARY (opening page of the report) ───────────────────
    def generate_executive_summary(self, domain_context: dict, ml_results: dict, cleaning_logs, dataset_name: str) -> str:
        """Writes the opening executive summary, grounded in the actual
        industry context, cleaning results, and best model score already
        computed elsewhere in the pipeline."""
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

    # ── CLEANING NARRATIVE (explains Section 2 of the PDF) ───────────────
    def generate_cleaning_narrative(self, cleaning_logs) -> str:
        """Turns the raw cleaning decision log (produced by CleaningAgent
        and IngestionAgent) into a flowing paragraph explaining what was
        done and why."""
        logs_str = json.dumps(cleaning_logs[:10] if isinstance(cleaning_logs, list) else cleaning_logs, default=str)
        prompt = f"""Write a highly detailed, 8-10 sentence narrative elaborating on the data cleaning decisions.
CLEANING LOG: {logs_str}
Mention key actions taken and provide descriptive justifications for why they were necessary. Be highly descriptive. Professional tone. No headers. Return ONLY text."""
        return self._call_llm(prompt, "Summarize cleaning.")

    # ── MODEL INTERPRETATION (explains Section 4 of the PDF) ─────────────
    def generate_ml_interpretation(self, ml_results: dict, domain_context: dict) -> str:
        """Explains the ML leaderboard and feature importance results in
        plain business language. Returns an empty string if no models
        were trained yet, so this section can be skipped cleanly."""
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

    # ── CONCLUSIONS AND RECOMMENDATIONS (closing section of the PDF) ─────
    def generate_conclusions(self, domain_context: dict, ml_results: dict, saved_queries: list) -> str:
        """Writes the closing conclusions/recommendations paragraph,
        drawing together the domain context, best model result, and any
        natural-language questions the user chose to save into the report."""
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
