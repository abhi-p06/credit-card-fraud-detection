"""
CreditGuard — Credit Card Fraud Detection using SVM
====================================================
Training Pipeline (train.py)

Dataset:
  Credit Card Fraud Detection Dataset 2023
  Source: Kaggle (nelgiriyewithana/credit-card-fraud-detection-dataset-2023)
  Total: 568,630 transactions (approximately 50% Class 0, 50% Class 1)

This script handles:
  1. Loading the 2023 dataset
  2. Data exploration (shape, missing values, duplicates, class distribution)
  3. Preprocessing:
     - Remove 'id' column
     - Feature/target split (X: V1-V28, Amount; y: Class)
     - Stratified 80/20 train/test split (random_state=42)
     - StandardScaler applied to numerical features (fitted on train only)
     - No undersampling, no SMOTE (dataset is inherently class-balanced)
  4. Training SVM models (Linear and RBF kernels)
  5. Dynamic evaluation (Confusion Matrix, Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC)
  6. Saving trained models and results

Usage:
    python train.py
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
import joblib


# ── Configuration ─────────────────────────────────────────────
DATA_PATH = "data/creditcard_2023.csv"
MODELS_DIR = "models"
RANDOM_STATE = 42
TEST_SIZE = 0.2


# ==============================================================
# STEP 1 : Load the dataset
# ==============================================================
def load_data(path=DATA_PATH):
    """Load the Credit Card Fraud Detection Dataset 2023 CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download from: https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023\n"
            "Place the file at: data/creditcard_2023.csv"
        )
    df = pd.read_csv(path)
    return df


# ==============================================================
# STEP 2 : Explore the dataset
# ==============================================================
def explore_data(df):
    """Print basic information about the dataset."""
    print("=" * 60)
    print("  DATASET OVERVIEW — Credit Card Fraud 2023")
    print("=" * 60)
    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    print("Columns:")
    print(list(df.columns))
    print()

    print("First 5 rows:")
    print(df.head())
    print()

    # Missing values check
    missing = df.isnull().sum().to_dict()
    total_missing = sum(missing.values())
    print(f"Missing values check: Total = {total_missing}")
    if total_missing > 0:
        for col, cnt in missing.items():
            if cnt > 0:
                print(f"  {col}: {cnt}")
    else:
        print("  ✓ Zero missing values found across all columns.")
    print()

    # Duplicates check (excluding 'id')
    feature_cols = [c for c in df.columns if c != "id"]
    dup_count = int(df.duplicated(subset=feature_cols).sum())
    print(f"Duplicate rows check (excluding 'id'): {dup_count:,} duplicate(s) found.")
    print()

    # Class distribution
    class_counts = df["Class"].value_counts().to_dict()
    print("Class distribution:")
    print(f"  Legitimate (0): {class_counts.get(0, 0):,} ({class_counts.get(0, 0)/len(df)*100:.2f}%)")
    print(f"  Fraud      (1): {class_counts.get(1, 0):,} ({class_counts.get(1, 0)/len(df)*100:.2f}%)")
    print("  ✓ Dataset is inherently class-balanced (~50% Class 0, ~50% Class 1).")
    print("  ✓ No artificial undersampling or SMOTE required.")
    print()


# ==============================================================
# STEP 3 : Preprocess the dataset
# ==============================================================
def preprocess(df):
    """
    Preprocessing steps:
      1. Drop the 'id' column (transaction identifier)
      2. Separate features (X: V1-V28, Amount) and target (y: Class)
      3. Stratified 80/20 train/test split (random_state=42)
      4. Standardize numerical features using StandardScaler
         - fit on training data only (no data leakage)
         - transform both train and test sets
      5. No undersampling or artificial balancing
    """
    print("=" * 60)
    print("  PREPROCESSING")
    print("=" * 60)

    # 3.1 Drop 'id' column
    df_clean = df.drop(columns=["id"]) if "id" in df.columns else df
    print(f"\n✓ Removed 'id' column. Remaining columns: {df_clean.shape[1]}")

    # 3.2 Separate features (X) and target (y)
    X = df_clean.drop(columns=["Class"])
    y = df_clean["Class"]
    print(f"Features (X): {X.shape[0]:,} samples × {X.shape[1]} features")
    print(f"Target   (y): {y.shape[0]:,} samples (0: legitimate, 1: fraud)")

    # 3.3 Train/test split (80/20, stratified, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nTrain set (80%): {X_train.shape[0]:,} samples")
    print(f"  Legitimate (0): {(y_train == 0).sum():,} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")
    print(f"  Fraud      (1): {(y_train == 1).sum():,} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")

    print(f"\nTest set (20%):  {X_test.shape[0]:,} samples")
    print(f"  Legitimate (0): {(y_test == 0).sum():,} ({(y_test == 0).sum()/len(y_test)*100:.1f}%)")
    print(f"  Fraud      (1): {(y_test == 1).sum():,} ({(y_test == 1).sum()/len(y_test)*100:.1f}%)")

    # 3.4 StandardScaler
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    print("\n✓ Standardized numerical features with StandardScaler")
    print("  (fitted strictly on training set to prevent data leakage)")
    print("  ✓ No undersampling applied — utilizing balanced natural dataset.")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ==============================================================
# STEP 4 : Save preprocessed data
# ==============================================================
def save_preprocessed(X_train, X_test, y_train, y_test, scaler):
    """Save the preprocessed data and scaler."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump({
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }, os.path.join(MODELS_DIR, "preprocessed_data.joblib"))

    print(f"\n✓ Scaler and preprocessed data saved to '{MODELS_DIR}/'")


# ==============================================================
# STEP 5 : Train SVM models
# ==============================================================
def train_svm(X_train, y_train, kernel="linear"):
    """
    Train a Support Vector Machine classifier.

    Parameters:
        X_train : training features (balanced)
        y_train : training labels (balanced)
        kernel  : 'linear' or 'rbf'

    Returns:
        model      : the trained SVC model
        train_time : training duration (seconds)
    """
    print(f"\n⏳ Training {kernel.upper()} SVM ...")

    if kernel == "linear":
        model = SVC(kernel="linear", cache_size=1000)
    else:
        model = SVC(kernel="rbf", C=1.0, gamma="scale", cache_size=1000)

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    print(f"✓ {kernel.upper()} SVM trained in {train_time:.1f} seconds")
    return model, train_time


# ==============================================================
# STEP 6 : Evaluate a trained model
# ==============================================================
def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained SVM model on the test set dynamically.
    """
    print(f"\n{'='*60}")
    print(f"  {model_name} — Dynamic Evaluation Results")
    print(f"{'='*60}")

    # Generate predictions dynamically
    y_pred = model.predict(X_test)
    y_scores = model.decision_function(X_test)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\nConfusion Matrix:")
    print(f"                 Predicted Legit  Predicted Fraud")
    print(f"  Actual Legit   {tn:>14,}  {fp:>15,}")
    print(f"  Actual Fraud   {fn:>14,}  {tp:>15,}")

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_scores))
    pr_auc = float(average_precision_score(y_test, y_scores))

    print(f"\n  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"  Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  PR-AUC:    {pr_auc:.4f}")

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    # Calculate curve points
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_scores)

    return {
        "name": model_name,
        "y_pred": y_pred,
        "y_scores": y_scores,
        "confusion_matrix": cm,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": prec_curve,
        "recall_curve": rec_curve,
    }


# ==============================================================
# STEP 7 : Save trained models and results
# ==============================================================
def save_models(linear_model, rbf_model, linear_results, rbf_results,
                X_test, y_test, scaler):
    """Save the trained models and all evaluation results."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(linear_model, os.path.join(MODELS_DIR, "linear_svm.joblib"))
    joblib.dump(rbf_model, os.path.join(MODELS_DIR, "rbf_svm.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump({
        "linear": linear_results,
        "rbf": rbf_results,
        "X_test": X_test,
        "y_test": y_test,
    }, os.path.join(MODELS_DIR, "results.joblib"))

    print(f"\n✓ Models and results saved to '{MODELS_DIR}/'")
    print(f"  - linear_svm.joblib")
    print(f"  - rbf_svm.joblib")
    print(f"  - scaler.joblib")
    print(f"  - results.joblib")


# ==============================================================
# Main
# ==============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  CreditGuard — Credit Card Fraud Detection using SVM")
    print("  Dataset 2023 Pipeline")
    print("=" * 60)
    print()

    # Step 1: Load
    df = load_data()

    # Step 2: Explore
    explore_data(df)

    # Step 3: Preprocess
    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    # Step 4: Save preprocessed data
    save_preprocessed(X_train, X_test, y_train, y_test, scaler)

    # Step 5: Train SVM models
    linear_model, linear_time = train_svm(X_train, y_train, kernel="linear")
    rbf_model, rbf_time = train_svm(X_train, y_train, kernel="rbf")

    # Step 6: Evaluate both models on the untouched test set
    linear_results = evaluate_model(linear_model, X_test, y_test, "Linear SVM")
    linear_results["train_time"] = linear_time

    rbf_results = evaluate_model(rbf_model, X_test, y_test, "RBF SVM")
    rbf_results["train_time"] = rbf_time

    # Step 7: Save models and results
    save_models(linear_model, rbf_model, linear_results, rbf_results,
                X_test, y_test, scaler)

    print("\n" + "=" * 60)
    print("  ✓ Pipeline complete! Models trained and evaluated.")
    print("=" * 60)
