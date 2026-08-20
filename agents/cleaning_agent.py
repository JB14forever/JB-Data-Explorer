# ==================================================================================
#  FILE: agents/cleaning_agent.py
# ==================================================================================
#  WHAT THIS FILE DOES (in plain English):
#  This is "Agent 3" in the pipeline. Once the obviously useless columns
#  have been removed (by ingestion_agent.py), this agent tidies up
#  everything that remains: messy column names, inconsistent text, missing
#  values, duplicate rows, and extreme/outlier numbers. Like the ingestion
#  agent, every decision here follows a FIXED, repeatable rule rather than
#  an AI judgement call — this is the "deterministic computation" half of
#  the platform's design (see Research Paper, Section 3.4).
# ==================================================================================

import pandas as pd
import numpy as np
import re
from scipy.stats import skew


class CleaningAgent:
    """
    Cleans the dataset utilizing the rigorous 12-Step Architecture:
    1. Understand limits, 2. Handle missing, 3. Duplicate removal
    4. Type fixing, 5. Standardization, 6. Outliers
    7. Text cleaning, 8. Inconsistency, 9. Validation.
    """

    # ── STEP 5: STANDARDISE COLUMN NAMES ─────────────────────────────────
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 5: Standardize Column Names.
        Turns messy headers like "Customer ID!" or "Monthly Charges" into a
        consistent, code-friendly format like "customer_id" / "monthly_charges"
        — lowercase, spaces become underscores, and any stray punctuation
        is stripped out."""
        df_clean = df.copy()
        new_cols = []
        for col in df_clean.columns:
            # Lowercase, replace spaces with _, remove non-alphanumeric (except _)
            c = str(col).lower().strip()
            c = c.replace(' ', '_')
            c = re.sub(r'[^a-z0-9_]', '', c)
            new_cols.append(c)
        df_clean.columns = new_cols
        return df_clean

    # ── STEPS 4, 7 & 8: FIX DATA TYPES AND CLEAN TEXT ────────────────────
    def fix_data_types_and_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 4 & 7: Convert to correct types and clean text strings.
        Some columns look like text but are actually dates stored as text
        (e.g. "2023-01-15") — this step detects and converts those to real
        dates. Any column that's still plain text afterwards gets trimmed
        of stray whitespace and lower-cased, so that "Male", "male " and
        "MALE" are all treated as the same value (Step 8: consistency)."""
        df_clean = df.copy()

        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Try converting to datetime first
                sample = df_clean[col].dropna().head(20).astype(str)
                # Quick heuristic to avoid casting pure text to datetime arbitrarily
                if sample.str.match(r'^\d{4}-\d{2}-\d{2}|^\d{2}/\d{2}/\d{4}').any():
                    try:
                        df_clean[col] = pd.to_datetime(df_clean[col], errors='ignore')
                    except Exception:
                        pass

                # If it's still object, clean the text
                if df_clean[col].dtype == 'object':
                    df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
                    # Step 8 handle inconsistencies like "male " -> "male", already largely fixed by strip/lower.
        return df_clean

    # ── STEP 2: HANDLE MISSING VALUES ────────────────────────────────────
    def handle_missing(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Step 2: Missing Values.
        The rule is simple and explained to the user in plain language:
          - If MORE than 30% of a column's values are missing, the whole
            column is dropped — too much is unknown to trust it.
          - If 30% or LESS is missing, the gaps are filled in ("imputed")
            using a sensible statistic instead of just deleting rows:
              * Numeric column, roughly bell-shaped   -> fill with the MEAN
              * Numeric column, skewed/lopsided        -> fill with the MEDIAN
                (median is less thrown off by extreme values than mean)
              * Date column                            -> carry the nearest
                known date forward/backward
              * Text/category column                   -> fill with the MODE
                (the single most common value)

        Returns the cleaned dataframe plus a dictionary explaining, for
        every affected column, exactly what was done and why — this
        justification text is what shows up in the app's audit log.
        """
        df_clean = df.copy()
        total_rows = len(df_clean)
        dropped_cols = {}

        for col in df_clean.columns:
            null_count = df_clean[col].isnull().sum()
            if null_count == 0:
                continue  # nothing missing in this column — skip it

            null_pct = (null_count / total_rows) * 100

            if null_pct > 30:
                df_clean = df_clean.drop(columns=[col])
                dropped_cols[col] = f"Dropped column entirely due to severe missingness ({null_pct:.1f}% missing, which exceeds the 30% reliability threshold)."
            else:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    # skew() measures how lopsided the distribution is;
                    # a value near 0 means roughly symmetric ("normal"),
                    # a large positive/negative value means it's skewed.
                    c_skew = skew(df_clean[col].dropna())
                    if pd.isna(c_skew) or abs(c_skew) > 0.5:
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                        dropped_cols[col] = f"Imputed {null_pct:.1f}% missing values using the Median. Justification: Distribution is highly skewed, making median more robust than mean."
                    else:
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                        dropped_cols[col] = f"Imputed {null_pct:.1f}% missing values using the Mean. Justification: Distribution is relatively normal."
                elif pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].ffill().bfill()
                    dropped_cols[col] = f"Imputed {null_pct:.1f}% missing values using Forward/Backward Fill to maintain temporal continuity."
                else:
                    mode_s = df_clean[col].mode()
                    if not mode_s.empty:
                        df_clean[col] = df_clean[col].fillna(mode_s[0])
                        dropped_cols[col] = f"Imputed {null_pct:.1f}% missing categorical values using the Mode (most frequent value)."

        return df_clean, dropped_cols

    # ── STEPS 6 & 9: OUTLIER HANDLING AND RANGE VALIDATION ───────────────
    def handle_outliers_and_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 6 & 9: Winsorize outliers and validate ranges.
        "Winsorizing" means capping extreme values instead of deleting
        them — any value far outside the normal range gets pulled in to
        the nearest reasonable boundary rather than removed, which avoids
        losing whole rows of otherwise-useful data.

        The standard statistical method used here is the IQR (Interquartile
        Range) rule: values more than 1.5x the IQR below the 25th percentile
        or above the 75th percentile are treated as outliers and clipped.

        A small extra safety check also prevents obviously impossible
        negative values in columns that should never be negative (age,
        salary, price, amount).
        """
        df_out = df.copy()
        numeric_cols = df_out.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            # Range validation: absolute heuristics.
            # If the column name suggests it should never be negative
            # (age, salary, price, amount), clip anything below zero.
            import math
            if 'age' in col or 'salary' in col or 'price' in col or 'amount' in col:
                df_out[col] = df_out[col].clip(lower=0)

            # Outlier Handling via IQR
            q1 = df_out[col].quantile(0.25)
            q3 = df_out[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            df_out[col] = np.where(df_out[col] < lower_bound, lower_bound, df_out[col])
            df_out[col] = np.where(df_out[col] > upper_bound, upper_bound, df_out[col])

        return df_out

    # ── STEP 3: REMOVE DUPLICATE ROWS ────────────────────────────────────
    def remove_duplicates(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Step 3: Remove Duplicate Rows.
        A duplicate row is one that is identical, cell-for-cell, to another
        row already in the dataset — these are removed so they don't
        artificially inflate patterns during analysis or modelling."""
        initial_len = len(df)
        df_dedup = df.drop_duplicates().copy()
        removed = initial_len - len(df_dedup)
        return df_dedup, removed

    # ── ORCHESTRATOR: RUNS ALL THE ABOVE STEPS IN ORDER ──────────────────
    def clean(self, df: pd.DataFrame, progress_bar=None, start_pct=0, end_pct=100) -> tuple[pd.DataFrame, dict, int]:
        """
        Orchestrates the 12-Step pipeline up to step 9, calling each of the
        cleaning functions above in the correct order and, if a Streamlit
        progress bar was passed in, updating it after every step so the
        user can watch the pipeline progress in the UI.
        Encoding and Scaling (steps 10-11) are delegated to
        TransformationAgent (see agents/transformation_agent.py), since
        those steps specifically prepare data for machine learning rather
        than "cleaning" it in the general sense.
        """
        step_size = (end_pct - start_pct) / 5
        current_pct = start_pct

        # Step 5: Standardization
        df_step = self.standardize_columns(df)
        if progress_bar: current_pct += step_size; progress_bar.progress(int(current_pct))

        # Step 4, 7, 8: Type fixing and Text standardization
        df_step = self.fix_data_types_and_text(df_step)
        if progress_bar: current_pct += step_size; progress_bar.progress(int(current_pct))

        # Step 3: Duplicate removal
        df_step, duplicates_removed = self.remove_duplicates(df_step)
        if progress_bar: current_pct += step_size; progress_bar.progress(int(current_pct))

        # Step 2: Missing Values
        df_step, missing_drops = self.handle_missing(df_step)
        if progress_bar: current_pct += step_size; progress_bar.progress(int(current_pct))

        # Step 6 & 9: Outliers & Range Validation
        df_step = self.handle_outliers_and_ranges(df_step)
        if progress_bar: current_pct += step_size; progress_bar.progress(int(end_pct))

        return df_step, missing_drops, duplicates_removed
