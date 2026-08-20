# ==================================================================================
#  FILE: agents/ml_agent.py
# ==================================================================================
#  WHAT THIS FILE DOES (in plain English):
#  This is "Agent 5" — the predictive modelling engine. Once the dataset
#  has been cleaned and transformed into numbers, this agent automatically
#  trains SEVERAL different machine learning algorithms on it at once,
#  compares how well each one performs, and reports back the winner along
#  with which columns ("features") most influenced its predictions.
#
#  Rather than a black box that just hands back "the model", the platform
#  deliberately builds and shows a full leaderboard — this transparency is
#  what let the research evaluation quickly spot and diagnose an unusually
#  strong result during testing (see Research Paper, Section 5.2 and
#  Evaluation, Section 6.1). A concrete, planned strengthening of this
#  agent is an automatic pre-modelling check that flags any candidate
#  feature suspiciously derived from the prediction target itself, before
#  training even begins — turning that diagnostic strength into a built-in
#  safeguard (see Client Report, Section 5.1, Recommendation 1).
# ==================================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, root_mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


class MLAgent:
    """
    Handles automated machine learning model selection, training, and evaluation.
    Now includes an expanded suite of algorithms and comparative leaderboards.
    """

    # ── STEP A: DECIDE WHAT KIND OF PROBLEM THIS IS ──────────────────────
    def detect_task(self, df: pd.DataFrame, target_col: str) -> str:
        """
        Looks at the column chosen as the prediction target and decides
        whether this is a CLASSIFICATION problem (predicting one of a
        limited set of categories, e.g. "will churn" / "won't churn") or a
        REGRESSION problem (predicting an open-ended numeric value, e.g.
        "monthly charges"). The rule: if the target is text/boolean, or a
        number with 20 or fewer distinct values, treat it as classification;
        otherwise treat it as regression.
        """
        target = df[target_col]
        dtype = target.dtype
        n_unique = target.nunique()

        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_bool_dtype(dtype) or n_unique <= 20:
            return 'classification'
        return 'regression'

    # ── STEP B: EXPLAIN WHICH COLUMNS MATTERED MOST ──────────────────────
    def get_feature_importance(self, model, feature_names: list) -> dict:
        """
        Extracts a ranked list of which input columns had the biggest
        influence on the winning model's predictions. Tree-based models
        (Random Forest, XGBoost, Decision Tree) expose this natively via
        `.feature_importances_`; linear models (Logistic/Linear Regression)
        expose it via their learned coefficients instead. Returns the top
        10 most influential features, most important first.

        This output is exactly what allowed the research evaluation to
        spot that two columns were dominating the model's predictions —
        demonstrating why this step is kept front-and-centre in the UI
        rather than buried, and why it is the foundation for the planned
        automatic leakage check described above.
        """
        importances = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)

        if importances is not None:
            feat_imp = {feat: float(imp) for feat, imp in zip(feature_names, importances)}
            sorted_feats = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True)[:10])
            return sorted_feats
        return {}

    # ── MAIN ENTRY POINT: TRAIN AND COMPARE ALL MODELS ───────────────────
    def train(self, df: pd.DataFrame, target_col: str) -> dict:
        """
        Splits data, trains competing tree-based and linear algorithms,
        and extracts a leaderboard of the results across multiple metrics.

        Step by step:
          1. Work out if this is classification or regression (detect_task).
          2. Keep only numeric/boolean columns as inputs — text columns
             would already have been converted to numbers by
             TransformationAgent before this function is ever called.
          3. Split the data 80% for training / 20% for testing, using a
             fixed random seed (random_state=42) so results are repeatable.
          4. Train every candidate algorithm one at a time. If any single
             algorithm fails for some reason, it's skipped rather than
             stopping the whole sweep (see the `except Exception: continue`
             lines below) — this keeps the pipeline resilient.
          5. Record every model's scores into a "leaderboard" table and
             keep track of whichever model performed best.
          6. Extract feature importance from the best model.
        """
        from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, r2_score

        task_type = self.detect_task(df, target_col)
        df_clean = df.dropna(subset=[target_col]).copy()

        # Only numeric/boolean columns can be fed into these algorithms.
        features = df_clean.drop(columns=[target_col])
        features = features.select_dtypes(include=[np.number, bool])

        X = features
        y = df_clean[target_col]

        # Fixed random_state=42 makes the train/test split reproducible —
        # running the pipeline again on the same data gives the same split.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        leaderboard = []
        best_model_name = ""
        best_model_obj = None
        best_metric_val = None
        metric_name = "Primary Score"

        if task_type == 'classification':
            metric_name = 'F1-Score'
            best_metric_val = -1.0

            # ROC-AUC is only meaningful for a two-outcome (binary) target.
            is_binary = len(np.unique(y)) == 2

            # The five candidate algorithms trained and compared for every
            # classification problem, spanning simple/interpretable models
            # through to more complex ensemble methods.
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Support Vector Classifier': SVC(probability=True, random_state=42)
            }

            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                    roc = "N/A"
                    if is_binary and hasattr(model, "predict_proba"):
                        y_prob = model.predict_proba(X_test)[:, 1]
                        roc = roc_auc_score(y_test, y_prob)

                    leaderboard.append({'Model': name, 'Accuracy': acc, 'F1-Score': f1, 'ROC-AUC': roc})

                    # F1-Score (weighted) is the primary metric used to pick
                    # the "winning" model, since it balances precision and
                    # recall even when the target classes are imbalanced.
                    if f1 > best_metric_val:
                        best_metric_val = f1
                        best_model_name = name
                        best_model_obj = model
                except Exception:
                    # If one algorithm errors out, skip it and keep going
                    # with the rest of the sweep instead of failing entirely.
                    continue

        else:
            metric_name = 'RMSE'
            best_metric_val = float('inf')

            # The five candidate algorithms trained and compared for every
            # regression problem.
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'XGBoost': XGBRegressor(random_state=42),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'Linear Regression': LinearRegression(),
                'Support Vector Regressor': SVR()
            }

            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    rmse = root_mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    leaderboard.append({'Model': name, 'RMSE': rmse, 'MAE': mae, 'R²': r2})

                    # For regression, RMSE (lower is better) picks the winner.
                    if rmse < best_metric_val:
                        best_metric_val = rmse
                        best_model_name = name
                        best_model_obj = model
                except Exception:
                    continue

        # Sort the leaderboard so the best-performing model appears first —
        # highest F1 first for classification, lowest RMSE first for regression.
        if task_type == 'classification':
            leaderboard = sorted(leaderboard, key=lambda x: x['F1-Score'], reverse=True)
        else:
            leaderboard = sorted(leaderboard, key=lambda x: x['RMSE'], reverse=False)

        feature_importance = self.get_feature_importance(best_model_obj, X.columns.tolist())

        return {
            'task_type': task_type,
            'best_model_name': best_model_name,
            'best_model_obj': best_model_obj,
            'best_metric_value': round(float(best_metric_val), 4) if best_metric_val not in (-1.0, float('inf')) else 0.0,
            'metric_name': metric_name,
            'feature_importance': feature_importance,
            'leaderboard': leaderboard
        }
