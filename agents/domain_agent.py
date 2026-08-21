# ==================================================================================
#  FILE: agents/domain_agent.py
# ==================================================================================
#  Agent 2 in the pipeline, and the first place AI gets used. Once the
#  useless columns are stripped out, this looks at the column names and a
#  small sample of rows and asks the AI to work out, in business terms:
#  what industry is this data from, what's it trying to predict, and how
#  should that prediction be measured?
#
#  Important: this never calculates any statistics itself, it only
#  interprets context. If the AI is unavailable, a simple rule-based
#  guess is used instead so the app keeps working either way.
# ==================================================================================

import json
from utils.llm_client import get_llm_client, LLM_MODEL, is_llm_available

class DomainAgent:
    """
    Leverages OpenAI to understand industry context and recommend configurations.
    """

    def __init__(self):
        self.client = get_llm_client()
        self.available = self.client is not None

    def analyze_context(self, schema: dict, sample_data: list) -> dict:
        """
        Takes schema and row sample, asks LLM to identify the dataset.
        Returns JSON structure: {
            "industry": string,
            "target_variable": string,
            "evaluation_metric": string,
            "business_summary": string,
            "suggested_queries": list[string]
        }
        """
        if not self.available:
            return self._fallback_context(schema)

        # the instruction sent to the AI, keeping the format strict so it
        # always comes back parseable
        system_prompt = """
        You are a Principal Data Scientist advising an enterprise analytics platform.
        You will be provided with a JSON schema of a dataset and a small sample of rows.

        Analyze the data and output ONLY a raw JSON string matching exactly this format (no markdown tags):
        {
            "industry": "Extracted Industry Context (e.g. Healthcare, Finance...)",
            "problem_type": "One of: Classification, Regression, Clustering, Time Series, Recommendation, NLP, Computer Vision",
            "target_variable": "The exact name of the column mathematically best suited to be the ML prediction target.",
            "evaluation_metrics": "Select metrics from the rule table based on the problem_type.",
            "business_summary": "An elaborate 18-sentence domain context (strictly 3 paragraphs of exactly 6 sentences each) that provides a meaningful and relatable industry context to what this dataset is modeling.",
            "suggested_queries": ["Suggest 1 basic query", "Suggest 1 advanced insight query"]
        }

        RULE TABLE FOR METRICS:
        - Classification: Accuracy, F1, ROC-AUC
        - Regression: RMSE, MAE, R²
        - Clustering: Silhouette Score
        - Time Series: RMSE, MAPE
        - Recommendation: Precision@K, Recall@K
        - NLP: F1, BLEU
        - Computer Vision: Accuracy, mAP
        """

        # only the column schema and a few sample rows go to the AI,
        # never the full dataset, to keep the request small and private
        user_prompt = f"SCHEMA: {json.dumps(schema)}\nSAMPLE ROWS: {json.dumps(sample_data, default=str)}"

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1  # low temperature keeps answers consistent
            )
            raw = response.choices[0].message.content.strip()

            # the AI is asked for pure JSON but sometimes wraps it in
            # markdown fences anyway, strip those before parsing
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.endswith("```"):
                raw = raw[:-3]

            return json.loads(raw)

        except Exception:
            # AI call failed, timed out, or returned something that isn't
            # valid JSON, fall back to the rule-based guess instead of
            # crashing the pipeline
            return self._fallback_context(schema)

    def _fallback_context(self, schema: dict) -> dict:
        """
        Rule-based backup for when the AI isn't available. Looks for
        column names that usually indicate a prediction target (e.g.
        "churn", "status", "price") and makes a best-effort guess from
        there. Keeps the platform usable with no AI key configured, just
        with simpler descriptions.
        """
        possible_targets = [c for c in schema.keys() if any(x in str(c).lower() for x in ['target', 'is_', 'status', 'churn', 'price', 'salary', 'outcome'])]
        target = possible_targets[0] if possible_targets else list(schema.keys())[-1]

        target_info = schema.get(target, {})
        is_numeric = target_info.get('dtype') == 'numeric'
        cardinality = target_info.get('cardinality', 100)

        if is_numeric and cardinality > 10:
            eval_metric = "RMSE, MAE, R² (Inferred Regression)"
            business_sum = "Predicting continuous numeric values based on historical trends."
        else:
            eval_metric = "Accuracy, F1-Score, ROC-AUC (Inferred Classification)"
            business_sum = "Categorizing or classifying outcomes based on the provided features."

        return {
            "industry": "General Business",
            "target_variable": target,
            "evaluation_metric": eval_metric,
            "business_summary": business_sum,
            "suggested_queries": [
                f"Show distribution of {target}.",
                f"How does {target} vary across different groups?"
            ]
        }
