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
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix, f1_score
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC



def build_pipeline():
    """
    Builds a scikit-learn Pipeline that performs:
    1. Feature selection using elastic net logistic regression.
    2. Final classification using logistic regression with balanced class weights.
    
    This pipeline was built using empirically tuned hyperparameters based on prior experimentation.
    """

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

def train_and_evaluate_logistic_regression_model(pipeline, X_train, y_train, X_test, y_test,X_blinded_test, train_IDs, test_IDs, blinded_test_IDs):
    """
    Train the logistic regression pipeline and evaluate its performance on test data.
    
    Parameters:
    - pipeline: scikit-learn Pipeline object including feature selection and logistic regression.
    - X_train, y_train: training features and labels.
    - X_test, y_test: testing features and labels.
    - X_blinded_test: features of the blinded test set (no labels).
    - train_IDs, test_IDs, blinded_test_IDs: pandas Series or list of IDs corresponding to each dataset.
    
    Returns:
    - pipeline: trained pipeline model.
    - selected_features: list of feature names selected by the feature selector.
    """
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    # Binary classification assumed
    acc = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_proba[:, 1])
    recall = recall_score(y_test, y_pred)  # Sensitivity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n Testing Metrics Logistic Regresssion:")
    print(f"Accuracy:     {acc:.4f}")
    print(f"AUROC:        {auroc:.4f}")
    print(f"Sensitivity:  {recall:.4f}")
    print(f"Specificity:  {specificity:.4f}")
    print(f"F1-score:     {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # How many features were selected
    selector = pipeline.named_steps['feature_selection']
    mask = selector.get_support()
    print(f"Number of features selected: {mask.sum()}")
    feature_names = X_train.columns
    selected_features = feature_names[mask]
    
    save_predictions(pipeline, X_train, train_IDs, "predictions_logreg_train_set.csv")
    save_predictions(pipeline, X_test, test_IDs, "predictions_logreg_test_set.csv")
    save_predictions(pipeline, X_blinded_test, blinded_test_IDs, "predictions_logreg_blinded_set.csv")
    
    return pipeline, selected_features

def get_top_features(X, y, top_n=30):
    """
    Identifies the top N most important features using a Random Forest classifier.

    Parameters:
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series or np.ndarray): Target variable.
    - top_n (int): Number of top features to return (default is 30).

    Returns:
    - List[str]: List of names of the top N most important features based on feature importance.
    
    Notes:
    - The feature importance is determined using a RandomForestClassifier with class balancing.
    """
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf.fit(X, y)
    importances = rf.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    top_features = importance_df.sort_values(by='importance', ascending=False).head(top_n)['feature'].tolist()
    return top_features


def train_final_rf_model(X, y):
    """
    Trains a Random Forest classifier using predefined hyperparameters optimized for generalization.

    Parameters:
    - X (pd.DataFrame): Feature matrix for training.
    - y (pd.Series or np.ndarray): Target labels.

    Returns:
    - RandomForestClassifier: Trained Random Forest model.

    Notes:
    - The model uses class weight balancing to handle imbalanced data.
    - Hyperparameters (e.g., depth, min samples, max features) were empirically tuned after experimentation.
    """    

    best_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=3,
        max_features=0.1,
        min_samples_leaf=15,
        min_samples_split=15,
        class_weight='balanced',
        random_state=42
    )
    """Random forest model and evaluate performance."""

    best_rf.fit(X, y)
    return best_rf

def train_and_evaluate_rf_model(X_train, y_train, X_test, y_test,X_blinded_test, train_IDs, test_IDs, blinded_test_IDs):
    """
    Trains a Random Forest classifier using the top features selected from training data,
    evaluates its performance on the test set, and saves prediction probabilities.

    Steps:
    1. Select top N (default 30) important features from training data using Random Forest feature importances.
    2. Train a tuned Random Forest model on the reduced feature set.
    3. Predict and evaluate metrics on the test set.
    4. Save prediction probabilities for train, test, and blinded test datasets.

    Parameters:
    - X_train (pd.DataFrame): Training feature matrix.
    - y_train (pd.Series or np.ndarray): Training labels.
    - X_test (pd.DataFrame): Test feature matrix.
    - y_test (pd.Series or np.ndarray): Test labels.
    - X_blinded_test (pd.DataFrame): Feature matrix for blinded test data (unlabeled).
    - train_IDs (pd.Series or list): Identifiers for training samples.
    - test_IDs (pd.Series or list): Identifiers for test samples.
    - blinded_test_IDs (pd.Series or list): Identifiers for blinded test samples.

    Returns:
    - dict: Dictionary with evaluation metrics: accuracy, AUROC, sensitivity, specificity, and F1-score.

    Prints:
    - Detailed classification report and confusion matrix for test set.
    - Evaluation metrics on the test set.
    """
    top_features = get_top_features(X_train, y_train)
    
    x_train_top_30 = X_train[top_features]
    x_test_top_30 = X_test[top_features]
    X_blinded_test_top_30 = X_blinded_test[top_features]
    
    model = train_final_rf_model(x_train_top_30, y_train)
    
    y_pred = model.predict(x_test_top_30)
    y_proba = model.predict_proba(x_test_top_30)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_proba)
    sensitivity = recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    f1 = f1_score(y_test, y_pred)

    print(f"\nTesting Metrics Random Forest:")
    print(f"Accuracy:      {acc:.4f}")
    print(f"AUROC:         {auroc:.4f}")
    print(f"Sensitivity:   {sensitivity:.4f}")
    print(f"Specificity:   {specificity:.4f}")
    print(f"F1-score:      {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    save_predictions(model, x_train_top_30, train_IDs, "predictions_randomforest_train_set.csv")
    save_predictions(model, x_test_top_30, test_IDs, "predictions_randomforest_test_set.csv")
    save_predictions(model, X_blinded_test_top_30, blinded_test_IDs, "predictions_randomforest_blinded_set.csv")
    
    return {
        'accuracy': acc,
        'auroc': auroc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1
    }
    
def train_svm_classifier(X_train, y_train, C=0.01):
    """
    Train a linear SVM classifier with balanced class weights.

    Parameters:
    - X_train: training features
    - y_train: training labels
    - C: regularization parameter - tuned by experimentation

    Returns:
    - trained SVC model
    """
    svm = SVC(kernel='linear', class_weight='balanced', C=C, probability=True, random_state=42)
    svm.fit(X_train, y_train)
    return svm


def train_and_evaluate_svm_model(X_train, y_train, X_test, y_test,X_blinded_test, train_IDs, test_IDs, blinded_test_IDs):
    """
    Evaluate a trained classification model and print metrics.

    Parameters:
    - model: trained classifier
    - input features
    - true labels
    - ids to datasets
    """
    model = train_svm_classifier(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_proba) if y_proba is not None else 'N/A'
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    print("\nTesting Metrics SVC:")
    print(f"Accuracy:     {acc:.4f}")
    print(f"AUROC:        {auroc if auroc == 'N/A' else f'{auroc:.4f}'}")
    print(f"Sensitivity:  {recall:.4f}")
    print(f"Specificity:  {specificity:.4f}")
    print(f"F1-score:     {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    save_predictions(model, X_train, train_IDs, "predictions_svm_train_set.csv")
    save_predictions(model, X_test, test_IDs, "predictions_svm_test_set.csv")
    save_predictions(model, X_blinded_test, blinded_test_IDs, "predictions_svm_blinded_set.csv")
    
    
    return {
        'accuracy': acc,
        'auroc': auroc,
        'sensitivity': recall,
        'specificity': specificity,
        'f1': f1
    }



def save_predictions(model, X, IDs, filename):
    """
    Generate predicted probabilities and save them with IDs to a CSV in ./results/.
    
    Parameters:
    - model: trained pipeline
    - X: input features
    - IDs: ID column (same order as X)
    - filename: output CSV filename (not full path)
    """
    # Ensure results directory exists
    os.makedirs('./results', exist_ok=True)

    # Predict probabilities
    y_proba = model.predict_proba(X)
    df_pred = pd.DataFrame(y_proba, columns=[f'prob_class_{i}' for i in range(y_proba.shape[1])])
    df_pred.insert(0, 'ID', IDs.reset_index(drop=True))

    # Build full path and save
    filepath = os.path.join('./results', filename)
    df_pred.to_csv(filepath, index=False)
    print(f"Saved predictions to {filepath}")

