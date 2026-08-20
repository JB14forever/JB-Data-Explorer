# ==================================================================================
#  FILE: agents/transformation_agent.py
# ==================================================================================
#  WHAT THIS FILE DOES (in plain English):
#  This is "Agent 4" in the pipeline. Machine learning models can only work
#  with numbers — they cannot read text categories like "Male"/"Female" or
#  raw calendar dates directly. This agent converts the already-cleaned
#  dataset (produced by cleaning_agent.py) into a fully numeric table ready
#  for model training, without changing what the data actually means.
# ==================================================================================

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


class TransformationAgent:
    """
    Transforms clean data into a format optimal for Machine Learning.
    Step 10 & 11: Encoding and Feature Scaling.
    """

    # ── DATE COLUMNS: BREAK THEM INTO NUMERIC PARTS ──────────────────────
    def handle_datetime_features(self, df: pd.DataFrame, schema: dict) -> tuple[pd.DataFrame, list]:
        """
        Extracts temporal features from datetimes and drops the original fields
        since ML models cannot digest raw datetime64 objects directly.
        For example, a "signup_date" column becomes three new numeric
        columns: signup_date_year, signup_date_month, signup_date_day.
        """
        df_feat = df.copy()
        dt_cols = [c for c in schema if schema[c]['dtype'] == 'datetime']
        dropped = []

        for col in dt_cols:
            if col in df_feat.columns:
                df_feat[f"{col}_year"] = df_feat[col].dt.year
                df_feat[f"{col}_month"] = df_feat[col].dt.month
                df_feat[f"{col}_day"] = df_feat[col].dt.day
                # After extracting numerics, toss original
                df_feat = df_feat.drop(columns=[col])
                dropped.append(col)

        return df_feat, dropped

    # ── CATEGORICAL COLUMNS: CONVERT TEXT CATEGORIES TO NUMBERS ──────────
    def encode(self, df: pd.DataFrame, schema: dict) -> pd.DataFrame:
        """
        Converts text/category columns into numeric form using one of two
        standard techniques, chosen automatically based on how many
        distinct categories the column has:
          - 10 or fewer categories -> "One-Hot Encoding": creates a separate
            0/1 column for each category (e.g. "contract_Month-to-month",
            "contract_One year", "contract_Two year"). This avoids implying
            any false ordering between categories.
          - More than 10 categories -> "Label Encoding": assigns each
            category a single integer code. Used for high-cardinality
            columns to avoid creating an unmanageable number of new columns.
        Boolean (True/False) columns are simply converted to 1/0.
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

    # ── NUMERIC COLUMNS: PUT THEM ON A COMMON SCALE ──────────────────────
    def scale(self, df: pd.DataFrame, schema: dict) -> pd.DataFrame:
        """
        Applies "Standard Scaling" to the original numeric columns, so that
        every numeric feature has a mean of 0 and a standard deviation of 1.
        This matters because many ML algorithms (e.g. Logistic Regression,
        SVM) can be unfairly biased toward columns with naturally larger
        numbers (like "annual_income" vs "num_dependants") unless everything
        is placed on the same scale first.
        """
        df_scaled = df.copy()

        # We only strictly scale original base numeric variables.
        # OHE and encoded labels often shouldn't be scaled, but StandardScaling them
        # isn't universally harmful for Trees/LinReg. To be precise, scale all purely continous:
        numeric_cols = [c for c in df_scaled.columns if c in schema and schema[c]['dtype'] == 'numeric']

        if numeric_cols:
            scaler = StandardScaler()
            df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

        return df_scaled

    # ── ORCHESTRATOR: RUNS ALL THE ABOVE STEPS IN ORDER ──────────────────
    def transform(self, df: pd.DataFrame, schema: dict) -> pd.DataFrame:
        """
        Orchestrates step 10 & 11 transformation sequence:
        dates -> numeric parts, then categories -> numeric codes,
        then numeric columns -> a common scale. Returns a dataset that is
        100% numeric and ready to be handed to MLAgent for model training.
        """
        # 1. Parse Datetimes to int features
        df_dt, dt_drops = self.handle_datetime_features(df, schema)

        # 2. Encode
        df_encoded = self.encode(df_dt, schema)

        # 3. Scale
        df_scaled = self.scale(df_encoded, schema)

        return df_scaled
