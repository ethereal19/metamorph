import os
import glob
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from feature_extraction import process_signal_sliding_window, FEATURE_NAMES, FS

INTENT_CLASSES = ["move", "stop", "select", "open"]


def load_dataset_from_csv_files(data_dir="."):
    """
    Finds and loads intent CSV files (e.g. move.csv, stop.csv, select.csv, open.csv).
    Applies sliding window feature extraction and artifact rejection.
    """
    all_features = []
    all_labels = []

    print("Loading CSV files and extracting Welch band power features...")
    for intent in INTENT_CLASSES:
        csv_file = os.path.join(data_dir, f"{intent}.csv")
        if not os.path.exists(csv_file):
            print(f" Warning: {csv_file} not found. Skipping...")
            continue

        print(f" Processing {csv_file}...")
        df = pd.read_csv(csv_file)
        
        # Extract features over sliding 1.5s windows with 50% overlap
        features_df = process_signal_sliding_window(
            df, fs=FS, window_sec=1.5, overlap_sec=0.75, artifact_multiplier=10.0
        )

        if len(features_df) > 0:
            all_features.append(features_df)
            all_labels.extend([intent] * len(features_df))
            print(f"   -> Extracted {len(features_df)} valid feature windows (artifacts rejected)")

    if not all_features:
        raise ValueError("No valid training CSV files found or all windows were rejected as artifacts!")

    X = pd.concat(all_features, ignore_index=True)
    y = np.array(all_labels)

    return X, y


def train_and_compare_models(X, y):
    """
    Trains and compares LDA, SVM (linear kernel), and Random Forest using 5-Fold Stratified CV.
    Returns the best performing model pipeline.
    """
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    models = {
        "LDA (Linear Discriminant Analysis)": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LinearDiscriminantAnalysis())
        ]),
        "SVM (Linear Kernel)": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="linear", probability=True, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    print("\n" + "=" * 75)
    print(" BCI MODEL PERFORMANCE COMPARISON (5-Fold Cross Validation)")
    print("=" * 75)
    print(f"{'Model Name':<38} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 75)

    best_score = -1.0
    best_model_name = None
    best_pipeline = None

    for name, pipeline in models.items():
        cv_results = cross_validate(pipeline, X, y_encoded, cv=cv, scoring=scoring)
        acc = np.mean(cv_results["test_accuracy"])
        prec = np.mean(cv_results["test_precision_macro"])
        rec = np.mean(cv_results["test_recall_macro"])
        f1 = np.mean(cv_results["test_f1_macro"])

        print(f"{name:<38} | {acc:.4f}     | {prec:.4f}     | {rec:.4f}     | {f1:.4f}")

        if acc > best_score:
            best_score = acc
            best_model_name = name
            best_pipeline = pipeline

    print("=" * 75)
    print(f"\n WINNING MODEL: {best_model_name} (Accuracy: {best_score * 100:.2f}%)")

    # Fit winning pipeline on full dataset
    best_pipeline.fit(X, y_encoded)

    # Save full bundle with joblib
    model_bundle = {
        "pipeline": best_pipeline,
        "label_encoder": label_encoder,
        "feature_names": list(X.columns),
        "model_name": best_model_name,
        "accuracy": best_score,
    }

    model_filename = "bci_model.joblib"
    joblib.dump(model_bundle, model_filename)
    print(f" Model bundle successfully saved to '{model_filename}' via joblib!")

    return model_bundle


def main():
    X, y = load_dataset_from_csv_files(data_dir=".")
    print(f"\nTotal Dataset Size: {len(X)} samples across classes: {np.unique(y)}")
    train_and_compare_models(X, y)


if __name__ == "__main__":
    main()
