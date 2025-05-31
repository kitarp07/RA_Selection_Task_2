# Task 2: Tabular Data Classification
## Overview

This project tackles a tabular data classification problem using three datasets: training, test, and blinded test sets. The goal is to build and evaluate machine learning models to accurately classify data points, following best practices in preprocessing, feature engineering, and hyperparameter tuning.

## Datasets
- Training set: Used to build and tune models.<br>
- Test set: Held-out data for unbiased model evaluation.<br>
- Blinded test set: Used to generate final class probability predictions, which will be evaluated against hidden ground truth.

## Modelling Approach
Implemented logistic regression, Random Forest, and SVM.
### Followed standard ML pipeline steps:
- Data preprocessing and cleaning<br>
- Feature engineering and selection<br>
- Hyperparameter tuning (using cross-validation)<br>

### Evaluation metrics reported include:
- Accuracy<br>
- AUROC (Area Under the ROC Curve)<br>
- Sensitivity (Recall / True Positive Rate)<br>
- Specificity (True Negative Rate)<br>
- F1-score<br>

## Deliverables
- Prediction CSV files for each dataset are located inside the `results/` folder.
- The detailed methodology report is named `report.pdf`.

## Install required packages with:
pip install -r requirements.txt

## Usage
Run the full processing pipeline with: python main.py


