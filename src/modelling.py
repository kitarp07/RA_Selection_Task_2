# modelling.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
import joblib  # Optional: to save model


def build_pipeline():
    """Build pipeline with fixed best hyperparameters."""
    feature_selector = SelectFromModel(
        LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            C=10,
            l1_ratio=0.9,
            max_iter=5000,
            class_weight='balanced',
            random_state=42
        )
    )

    clf = LogisticRegression(
        C=1,
        max_iter=5000,
        class_weight='balanced',
        random_state=42
    )

    pipeline = Pipeline([
        ('feature_selection', feature_selector),
        ('clf', clf)
    ])
    return pipeline

def train_and_evaluate_logistic_regression_model(pipeline, X_train, X_test, y_train, y_test):
    """Train pipeline and evaluate performance."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    f1 = f1_score(y_test, y_pred)
    print(f"Test F1-score: {f1:.4f}")

    # How many features were selected
    selector = pipeline.named_steps['feature_selection']
    mask = selector.get_support()
    print(f"Number of features selected: {mask.sum()}")

    return pipeline

