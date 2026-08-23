# ==================================================================================
#  FILE: agents/ml_agent.py
# ==================================================================================
#  Agent 5, the predictive modelling engine. Once the dataset is cleaned
#  and transformed into numbers, this trains several ML algorithms on it
#  at once, compares them, and reports the winner along with which
#  columns influenced its predictions the most.
#
#  Rather than just handing back "the model", the leaderboard is built
#  and shown in full, which is what made it possible to spot and diagnose
#  an unusually strong result during testing (see the README's "A finding
#  worth calling out" section). A useful next step for this agent is an
#  automatic pre-modelling check that flags any feature that looks
#  suspiciously derived from the target itself, before training even
#  starts, turning that diagnostic strength into a built-in safeguard.
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

    # ── decide what kind of problem this is ────────────────────────
    def detect_task(self, df: pd.DataFrame, target_col: str) -> str:
        """Looks at the target column and decides if this is
        classification (predicting one of a limited set of categories,
        e.g. will churn / won't churn) or regression (predicting an
        open-ended number, e.g. monthly charges). Text/boolean targets,
        or numeric targets with 20 or fewer distinct values, count as
        classification, everything else is regression."""
        target = df[target_col]
        dtype = target.dtype
        n_unique = target.nunique()

        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_bool_dtype(dtype) or n_unique <= 20:
            return 'classification'
        return 'regression'

    # ── explain which columns mattered most ─────────────────────────
    def get_feature_importance(self, model, feature_names: list) -> dict:
        """
        Pulls a ranked list of which columns had the biggest influence on
        the winning model's predictions. Tree-based models (Random
        Forest, XGBoost, Decision Tree) expose this through
        .feature_importances_, linear models (Logistic/Linear Regression)
        expose it through their coefficients instead. Returns the top 10,
        most important first.

        This output is exactly what let the target-leakage issue get
        caught during testing, so it stays front and centre in the UI
        rather than buried somewhere.
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

    # ── main entry point: train and compare all models ─────────────
    def train(self, df: pd.DataFrame, target_col: str) -> dict:
        """
        Splits data, trains competing tree-based and linear algorithms,
        and extracts a leaderboard of the results across multiple metrics.

        Steps flow:
          1. Work out classification vs regression (detect_task).
          2. Keep only numeric/boolean columns as inputs, text columns
             would already have been converted to numbers by
             TransformationAgent before this ever gets called.
          3. Split 80/20 train/test with a fixed random seed so results
             are repeatable.
          4. Train each candidate algorithm one at a time. If one fails,
             it just gets skipped rather than stopping the whole sweep.
          5. Record every model's scores into a leaderboard and keep
             track of the best one.
          6. Pull feature importance from the winner.
        """
        from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, r2_score

        task_type = self.detect_task(df, target_col)
        df_clean = df.dropna(subset=[target_col]).copy()

        # only numeric/boolean columns go into these algorithms
        features = df_clean.drop(columns=[target_col])
        features = features.select_dtypes(include=[np.number, bool])

        X = features
        y = df_clean[target_col]

        # fixed random_state=42 keeps the split reproducible across runs
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        leaderboard = []
        best_model_name = ""
        best_model_obj = None
        best_metric_val = None
        metric_name = "Primary Score"

        if task_type == 'classification':
            metric_name = 'F1-Score'
            best_metric_val = -1.0

            # ROC-AUC only really makes sense for a two-outcome target
            is_binary = len(np.unique(y)) == 2

            # the five candidates trained and compared for classification
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

                    # weighted F1 picks the winner, balances precision and
                    # recall even with imbalanced classes
                    if f1 > best_metric_val:
                        best_metric_val = f1
                        best_model_name = name
                        best_model_obj = model
                except Exception:
                    # skip this algorithm and keep going with the rest
                    continue

        else:
            metric_name = 'RMSE'
            best_metric_val = float('inf')

            # the five candidates trained and compared for regression
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

                    # lower RMSE picks the winner for regression
                    if rmse < best_metric_val:
                        best_metric_val = rmse
                        best_model_name = name
                        best_model_obj = model
                except Exception:
                    continue

        # sort the leaderboard so the best model shows up first
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
