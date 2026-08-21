# ==================================================================================
#  FILE: agents/cleaning_agent.py
# ==================================================================================
#  Agent 3 in the pipeline. Once the useless columns are gone (handled by
#  ingestion_agent.py), this tidies up whatever's left: messy column
#  names, inconsistent text, missing values, duplicate rows, and extreme
#  outlier numbers. Same as the ingestion agent, every decision here
#  follows a fixed rule rather than an AI judgement call.
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

    # ── standardise column names ─────────────────────────────────
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Turns messy headers like "Customer ID!" or "Monthly Charges"
        into a consistent format like "customer_id" / "monthly_charges",
        lowercase, spaces become underscores, stray punctuation stripped."""
        df_clean = df.copy()
        new_cols = []
        for col in df_clean.columns:
            c = str(col).lower().strip()
            c = c.replace(' ', '_')
            c = re.sub(r'[^a-z0-9_]', '', c)
            new_cols.append(c)
        df_clean.columns = new_cols
        return df_clean

    # ── fix data types and clean up text ────────────────────────
    def fix_data_types_and_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Some columns look like text but are actually dates stored as
        strings (e.g. "2023-01-15"), this catches and converts those.
        Whatever's still text after that gets trimmed and lowercased, so
        "Male", "male ", and "MALE" all end up treated as the same value."""
        df_clean = df.copy()

        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # try converting to datetime first
                sample = df_clean[col].dropna().head(20).astype(str)
                # quick check to avoid casting random text to datetime
                if sample.str.match(r'^\d{4}-\d{2}-\d{2}|^\d{2}/\d{2}/\d{4}').any():
                    try:
                        df_clean[col] = pd.to_datetime(df_clean[col], errors='ignore')
                    except Exception:
                        pass

                # if it's still object type after that, clean it up
                if df_clean[col].dtype == 'object':
                    df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()

        return df_clean

    # ── handle missing values ────────────────────────────────────
    def handle_missing(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        The rule here is simple:
          - More than 30% missing in a column: drop the whole column,
            too much is unknown to trust it.
          - 30% or less: fill the gaps with something sensible instead of
            deleting rows:
              * numeric, roughly bell-shaped   -> fill with the mean
              * numeric, skewed/lopsided        -> fill with the median
                (less thrown off by extreme values than the mean)
              * date column                     -> carry the nearest
                known date forward/backward
              * text/category column            -> fill with the mode
                (most common value)

        Returns the cleaned dataframe plus a dict explaining what got
        done to each affected column and why, this text ends up in the
        app's audit log.
        """
        df_clean = df.copy()
        total_rows = len(df_clean)
        dropped_cols = {}

        for col in df_clean.columns:
            null_count = df_clean[col].isnull().sum()
            if null_count == 0:
                continue

            null_pct = (null_count / total_rows) * 100

            if null_pct > 30:
                df_clean = df_clean.drop(columns=[col])
                dropped_cols[col] = f"Dropped column entirely due to severe missingness ({null_pct:.1f}% missing, which exceeds the 30% reliability threshold)."
            else:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    # skew near 0 means roughly symmetric, a big
                    # positive/negative value means it's lopsided
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

    # ── handle outliers and range validation ─────────────────────
    def handle_outliers_and_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Winsorizes outliers instead of deleting them, extreme values get
        capped at a reasonable boundary rather than being thrown out, so
        the rest of a row doesn't get lost over one bad number.

        Uses the standard IQR rule: anything more than 1.5x the
        interquartile range below the 25th percentile or above the 75th
        gets treated as an outlier and clipped.

        Also does a small sanity check on columns that should never be
        negative (age, salary, price, amount).
        """
        df_out = df.copy()
        numeric_cols = df_out.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            # if the column name suggests it should never go negative,
            # clip anything below zero
            import math
            if 'age' in col or 'salary' in col or 'price' in col or 'amount' in col:
                df_out[col] = df_out[col].clip(lower=0)

            # IQR outlier handling
            q1 = df_out[col].quantile(0.25)
            q3 = df_out[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            df_out[col] = np.where(df_out[col] < lower_bound, lower_bound, df_out[col])
            df_out[col] = np.where(df_out[col] > upper_bound, upper_bound, df_out[col])

        return df_out

    # ── remove duplicate rows ────────────────────────────────────
    def remove_duplicates(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """A duplicate row is one that matches another row, cell for
        cell, exactly. These get removed so they don't inflate patterns
        during analysis or modelling."""
        initial_len = len(df)
        df_dedup = df.drop_duplicates().copy()
        removed = initial_len - len(df_dedup)
        return df_dedup, removed

    # ── runs all the steps above in order ────────────────────────
    def clean(self, df: pd.DataFrame, progress_bar=None, start_pct=0, end_pct=100) -> tuple[pd.DataFrame, dict, int]:
        """
        Calls each of the cleaning functions above in the right order,
        and if a progress bar got passed in, updates it after each step
        so the user can watch it move. Encoding and scaling happen
        separately in TransformationAgent, since those are specifically
        about getting data ready for ML rather than "cleaning" it.
        """
        step_size = (end_pct - start_pct) / 5
        current_pct = start_pct

        # standardise column names
        df_step = self.standardize_columns(df)
        if progress_bar: current_pct += step_size; progress_bar.progress(int(current_pct))

        # fix types and clean text
        df_step = self.fix_data_types_and_text(df_step)
        if progress_bar: current_pct += step_size; progress_bar.progress(int(current_pct))

        # remove duplicates
        df_step, duplicates_removed = self.remove_duplicates(df_step)
        if progress_bar: current_pct += step_size; progress_bar.progress(int(current_pct))

        # handle missing values
        df_step, missing_drops = self.handle_missing(df_step)
        if progress_bar: current_pct += step_size; progress_bar.progress(int(current_pct))

        # handle outliers and ranges
        df_step = self.handle_outliers_and_ranges(df_step)
        if progress_bar: current_pct += step_size; progress_bar.progress(int(end_pct))

        return df_step, missing_drops, duplicates_removed
