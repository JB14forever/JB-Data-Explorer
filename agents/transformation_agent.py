# ==================================================================================
#  FILE: agents/transformation_agent.py
# ==================================================================================
#  Agent 4 in the pipeline. Machine learning models only work with
#  numbers, they can't read text categories like "Male"/"Female" or raw
#  calendar dates directly. This turns the already-cleaned dataset into a
#  fully numeric table ready for training, without changing what the data
#  actually means.
# ==================================================================================

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


class TransformationAgent:
    """
    Transforms clean data into a format optimal for Machine Learning.
    Step 10 & 11: Encoding and Feature Scaling.
    """

    # ── date columns: break them into numeric parts ──────────────
    def handle_datetime_features(self, df: pd.DataFrame, schema: dict) -> tuple[pd.DataFrame, list]:
        """Extracts year/month/day from any datetime columns and drops the
        original field, since models can't read a raw datetime object.
        For example "signup_date" becomes signup_date_year,
        signup_date_month, signup_date_day."""
        df_feat = df.copy()
        dt_cols = [c for c in schema if schema[c]['dtype'] == 'datetime']
        dropped = []

        for col in dt_cols:
            if col in df_feat.columns:
                df_feat[f"{col}_year"] = df_feat[col].dt.year
                df_feat[f"{col}_month"] = df_feat[col].dt.month
                df_feat[f"{col}_day"] = df_feat[col].dt.day
                df_feat = df_feat.drop(columns=[col])
                dropped.append(col)

        return df_feat, dropped

    # ── categorical columns: turn text categories into numbers ───
    def encode(self, df: pd.DataFrame, schema: dict) -> pd.DataFrame:
        """
        Converts categories into numbers using one of two approaches,
        picked automatically by how many distinct values a column has:
          - 10 or fewer categories: one-hot encoding, a separate 0/1
            column per category (e.g. "contract_Month-to-month",
            "contract_One year", "contract_Two year"). Avoids implying
            any false ordering between categories.
          - more than 10 categories: label encoding, each category gets
            a single integer code. Used for high-cardinality columns so
            it doesn't blow up into dozens of new columns.
        Boolean columns just get converted to 1/0.
        """
        df_encoded = df.copy()

        for col, meta in schema.items():
            if col not in df_encoded.columns:
                continue

            if meta['dtype'] == 'categorical':
                if meta['cardinality'] <= 10:
                    df_encoded = pd.get_dummies(df_encoded, columns=[col], drop_first=True)
                else:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

            elif meta['dtype'] == 'boolean':
                df_encoded[col] = df_encoded[col].astype(int)

        return df_encoded

    # ── numeric columns: put them on a common scale ───────────────
    def scale(self, df: pd.DataFrame, schema: dict) -> pd.DataFrame:
        """
        Applies standard scaling to the numeric columns so each ends up
        with a mean of 0 and a standard deviation of 1. Matters because
        algorithms like Logistic Regression or SVM can end up unfairly
        weighted toward whichever column has naturally bigger numbers
        (e.g. "annual_income" vs "num_dependants") unless they're all put
        on the same footing first.
        """
        df_scaled = df.copy()

        numeric_cols = [c for c in df_scaled.columns if c in schema and schema[c]['dtype'] == 'numeric']

        if numeric_cols:
            scaler = StandardScaler()
            df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

        return df_scaled

    # ── runs all the steps above in order ─────────────────────────
    def transform(self, df: pd.DataFrame, schema: dict) -> pd.DataFrame:
        """Dates become numeric parts, then categories become numeric
        codes, then numeric columns get put on a common scale. Returns a
        fully numeric dataset ready to hand to MLAgent for training."""
        df_dt, dt_drops = self.handle_datetime_features(df, schema)
        df_encoded = self.encode(df_dt, schema)
        df_scaled = self.scale(df_encoded, schema)
        return df_scaled
