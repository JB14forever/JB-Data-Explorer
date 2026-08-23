# ==================================================================================
#  FILE: agents/ingestion_agent.py
# ==================================================================================
#  Agent 1 in the pipeline, the first thing that touches a dataset the
#  user uploads. Three tasks:
#    1. Read the uploaded file (CSV or Excel) into a table.
#    2. Automatically drop columns that are not useful for analysis, like a
#       customer ID column, or a column where every row is identical.
#    3. Work out a rough Data Health Score describing how clean the file
#       already is.
#
#  Everything here is rule-based. Running it twice on the same
#  file gives the same result every time, which matters for keeping the
#  pipeline reproducible.
# ==================================================================================

import io
import csv
import pandas as pd


class IngestionAgent:
    """
    Handles the ingestion of raw data (CSV and Excel) and initial profiling.
    Detects and filters out useless primary keys and zero-variance columns.
    """

    # ── load the file ─────────────────────────────────────────────
    def load_data(self, file) -> pd.DataFrame:
        """Loads an uploaded file (CSV or Excel) into a DataFrame, pandas'
        name for a data table, basically a spreadsheet held in memory."""
        filename = file.name.lower()

        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            # openpyxl handles xlsx, xlrd handles xls, assuming xlsx here
            df = pd.read_excel(io.BytesIO(file.getvalue()))
            return df
        else:
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

    # drop the obviously useless columns 
    def filter_primary_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Removes three kinds of columns that carry no real analytical
        value and would just confuse later steps:
          1. Zero-variance columns, every row has the same value (e.g. a
             "Country" column where every customer is in the same place).
          2. ID-like columns by name, e.g. "customer_id", "uuid", that
             also have very high cardinality (almost every value unique),
             confirming they're actually identifiers.
          3. 100%-unique text columns, e.g. emails or reference numbers,
             where every row is different, so the column can't describe
             any pattern shared across rows.

        Returns the filtered dataframe plus a dict explaining exactly why
        each dropped column got dropped, this is what powers the audit
        log shown in the app.
        """
        df_filtered = df.copy()
        dropped = {}
        total_rows = len(df_filtered)

        if total_rows == 0:
            return df_filtered, dropped

        cols_to_drop = []

        for col in df_filtered.columns:
            # zero-variance check: does this column ever actually change?
            if df_filtered[col].nunique(dropna=False) <= 1:
                dropped[col] = "Zero-variance (All values are identical or completely missing)."
                cols_to_drop.append(col)
                continue

            # does the column name look like an ID?
            lname = str(col).lower()
            if lname in ['id', 'uuid', 'index'] or lname.endswith('_id') or lname.endswith(' id'):
                # only drop if the values back that up too, otherwise a
                # column just named "id" with repeated values might still
                # be meaningful
                if df_filtered[col].nunique() > (total_rows * 0.5):
                    dropped[col] = f"Detected as a primary key/ID by name ('{col}')."
                    cols_to_drop.append(col)
                    continue

            # 100% unique strings, emails, usernames, hashes, etc
            if df_filtered[col].dtype == 'object':
                if df_filtered[col].nunique() == total_rows:
                    dropped[col] = "100% unique string values (Likely a primary key or arbitrary identifier)."
                    cols_to_drop.append(col)

        if cols_to_drop:
            df_filtered = df_filtered.drop(columns=cols_to_drop)

        return df_filtered, dropped

    # ── schema profiling, used to describe the dataset to the AI later ──
    def infer_schema(self, df: pd.DataFrame) -> dict:
        """
        Builds a short profile for every column: data type (numeric,
        categorical, datetime, boolean), how many values are missing, and
        how many distinct values it has. Used both to display the dataset
        summary in the app and to give the AI enough context to guess the
        industry and likely target variable later on.
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

    # ── data health score, a simple 0-100 quality summary ────────────
    def compute_health_score(self, df: pd.DataFrame) -> float:
        """
        Produces a single 0-100 score for how technically clean the
        dataset is, this is not about whether the data is analytically
        appropriate for a specific modelling task, just whether it's
        complete and tidy.

        Three things pull the score down:
          - overall_null_pct: percentage of empty cells
          - dup_row_pct: percentage of rows that are exact duplicates
          - dominant_cols_pct: percentage of columns dominated (>95%) by
            one repeated value, which limits how useful they are
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

        # weights chosen so missingness matters most, then duplicates,
        # then dominant-value columns
        penalty = (overall_null_pct * 0.5) + (dup_row_pct * 0.3) + (dominant_cols_pct * 0.2)
        score = max(0.0, min(100.0, 100.0 - penalty))

        return round(score, 2)
