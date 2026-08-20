# ==================================================================================
#  FILE: agents/ingestion_agent.py
# ==================================================================================
#  WHAT THIS FILE DOES (in plain English):
#  This is "Agent 1" in the pipeline — the very first thing that happens to
#  a dataset the user uploads. Its three jobs are:
#    1. Read the uploaded file (CSV or Excel) into a table the app can use.
#    2. Automatically remove obviously useless columns (e.g. a customer ID
#       column, or a column where every row has the exact same value).
#    3. Work out a rough "Data Health Score" describing how clean the file
#       already is.
#
#  EVERYTHING IN THIS FILE IS DETERMINISTIC — meaning it uses fixed rules,
#  not AI. Running it twice on the same file always gives the same result,
#  which is important for the platform's "traceability" goal described in
#  the research paper (Section 3.4, "Justification of Choices").
# ==================================================================================

import io
import csv
import pandas as pd


class IngestionAgent:
    """
    Handles the ingestion of raw data (CSV and Excel) and initial profiling.
    Detects and filters out useless primary keys and zero-variance columns.
    """

    # ── STEP 1: LOAD THE FILE ────────────────────────────────────────────
    def load_data(self, file) -> pd.DataFrame:
        """
        Loads an uploaded file (CSV or Excel) into a DataFrame (pandas'
        name for a data table — think of it as a spreadsheet in memory).
        """
        filename = file.name.lower()

        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            # openpyxl handles xlsx, xlrd handles xls. We assume xlsx here.
            df = pd.read_excel(io.BytesIO(file.getvalue()))
            return df
        else:
            # Fallback to smart CSV sniffing.
            # Not every CSV file uses a comma as the separator (some use a
            # semicolon or a tab), so we "sniff" the first couple of
            # kilobytes of the file to guess the correct delimiter
            # automatically, rather than assuming it's always a comma.
            content = file.getvalue().decode('utf-8', errors='replace')
            sample = content[:2048]
            try:
                dialect = csv.Sniffer().sniff(sample)
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ','

            str_io = io.StringIO(content)
            df = pd.read_csv(str_io, sep=delimiter)
            return df

    # ── STEP 2: DROP OBVIOUSLY USELESS COLUMNS ───────────────────────────
    def filter_primary_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Removes three kinds of columns that carry no analytical value and
        would only confuse later modelling steps:
          1. Zero-variance columns — every row has the exact same value
             (e.g. a "Country" column where every customer is in Ireland).
          2. ID-like columns by name — e.g. "customer_id", "uuid" — that
             also have very high cardinality (almost every value is unique),
             confirming they are indeed identifiers rather than real data.
          3. 100%-unique text columns — e.g. emails or reference numbers —
             where every single row has a different value, meaning the
             column cannot describe any pattern shared across rows.

        Returns:
            df (pd.DataFrame): The dataset with those columns removed.
            dropped_reasoning (dict): A column-by-column log of exactly
                why each dropped column was removed — this is what powers
                the "Decision Justification" audit trail shown in the app.
        """
        df_filtered = df.copy()
        dropped = {}
        total_rows = len(df_filtered)

        if total_rows == 0:
            return df_filtered, dropped

        cols_to_drop = []

        for col in df_filtered.columns:
            # 1. Zero-Variance: does this column ever change value at all?
            if df_filtered[col].nunique(dropna=False) <= 1:
                dropped[col] = "Zero-variance (All values are identical or completely missing)."
                cols_to_drop.append(col)
                continue

            # 2. Heuristic ID naming: does the column NAME look like an ID?
            lname = str(col).lower()
            if lname in ['id', 'uuid', 'index'] or lname.endswith('_id') or lname.endswith(' id'):
                # Only drop if it actually looks like an ID (high cardinality) —
                # a column merely named "id" but with repeated values might
                # still carry useful information, so we double-check.
                if df_filtered[col].nunique() > (total_rows * 0.5):
                    dropped[col] = f"Detected as a primary key/ID by name ('{col}')."
                    cols_to_drop.append(col)
                    continue

            # 3. 100% Unique Strings (e.g. emails, usernames, hashes)
            if df_filtered[col].dtype == 'object':
                if df_filtered[col].nunique() == total_rows:
                    dropped[col] = "100% unique string values (Likely a primary key or arbitrary identifier)."
                    cols_to_drop.append(col)

        if cols_to_drop:
            df_filtered = df_filtered.drop(columns=cols_to_drop)

        return df_filtered, dropped

    # ── SCHEMA PROFILING (used to describe the dataset to the AI later) ──
    def infer_schema(self, df: pd.DataFrame) -> dict:
        """
        Builds a "schema" — a short profile for every column describing
        its data type (numeric / categorical / datetime / boolean), how
        many values are missing, and how many distinct values it has
        (cardinality). This profile is used both to display the dataset
        summary in the app and to give the AI enough context to guess the
        dataset's industry and likely prediction target (see domain_agent.py).
        """
        schema = {}
        total_rows = len(df)

        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            null_pct = (null_count / total_rows) * 100 if total_rows > 0 else 0

            if pd.api.types.is_numeric_dtype(df[col]):
                col_type = 'numeric'
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                col_type = 'datetime'
            elif pd.api.types.is_bool_dtype(df[col]):
                col_type = 'boolean'
            else:
                col_type = 'categorical'

            schema[col] = {
                'dtype': col_type,
                'null_count': null_count,
                'null_percentage': null_pct,
                'cardinality': int(df[col].nunique(dropna=False))
            }

        return schema

    # ── DATA HEALTH SCORE (a simple 0-100 quality summary) ───────────────
    def compute_health_score(self, df: pd.DataFrame) -> float:
        """
        Produces a single 0-100 score summarising the *technical* cleanliness
        of the dataset — NOT whether it is analytically appropriate for a
        given modelling task (that distinction is discussed in both the
        Research Paper, Section 6.1, and the Client Report, Section 4.5).

        The score is reduced ("penalised") by three factors:
          - overall_null_pct: what percentage of all cells are empty
          - dup_row_pct: what percentage of rows are exact duplicates
          - dominant_cols_pct: what percentage of columns are dominated
            (>95%) by a single repeated value, which limits their usefulness
        """
        if df.empty:
            return 0.0

        total_cells = df.shape[0] * df.shape[1]
        overall_null_pct = (df.isnull().sum().sum() / total_cells) * 100
        dup_row_pct = (df.duplicated().sum() / len(df)) * 100

        dominant_cols = 0
        for col in df.columns:
            if df[col].nunique() > 0:
                top_val_freq = df[col].value_counts(normalize=True).iloc[0] * 100
                if top_val_freq > 95:
                    dominant_cols += 1

        dominant_cols_pct = (dominant_cols / df.shape[1]) * 100

        # Weighted penalty: missingness matters most, then duplicates, then
        # dominant-value columns. Weights were chosen heuristically to
        # reflect their relative impact on downstream analysis quality.
        penalty = (overall_null_pct * 0.5) + (dup_row_pct * 0.3) + (dominant_cols_pct * 0.2)
        score = max(0.0, min(100.0, 100.0 - penalty))

        return round(score, 2)
