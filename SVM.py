# ============================================================
# SVM + LOGISTIC REGRESSION HYBRID GRADE PREDICTION
# ============================================================

import os
import joblib
import pandas as pd
import warnings
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.base import clone

warnings.filterwarnings("ignore")  # hide sklearn/numpy warnings so console output stays clean

# ============================================================
# SETTINGS
# ============================================================

ARTIFACTS_DIR = "artifacts"          # folder where inputs are read from and outputs are saved
TARGET = "Grade"                     # name of the label column
os.makedirs(ARTIFACTS_DIR, exist_ok=True)  # create the folder if it doesn't exist yet

print("\n" + "=" * 55)
print("SVM + Logistic Regression Hybrid Grade Prediction")
print("=" * 55)

# ============================================================
# LOAD DATA
# ============================================================

# Read pre-split train/test features and labels from CSV files
X_train = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_train_raw.csv"))
X_test = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_test_raw.csv"))
y_train_raw = pd.read_csv(os.path.join(ARTIFACTS_DIR, "y_train.csv"))[TARGET]
y_test_raw = pd.read_csv(os.path.join(ARTIFACTS_DIR, "y_test.csv"))[TARGET]

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# ============================================================
# FEATURE COLUMNS
# ============================================================

feature_path = os.path.join(ARTIFACTS_DIR, "feature_columns.joblib")

# If a previously saved feature-column order exists, reuse it so
# train/test/production data always line up the same way
if os.path.exists(feature_path):
    feature_columns = joblib.load(feature_path)
    X_train = X_train[feature_columns]
    X_test = X_test[feature_columns]
else:
    feature_columns = X_train.columns.tolist()

# ============================================================
# VALIDATION
# ============================================================

# Guard against accidentally leaking the label into the features
if TARGET in X_train.columns:
    raise ValueError("Grade exists inside X_train.")

if TARGET in X_test.columns:
    raise ValueError("Grade exists inside X_test.")

# Guard against train/test having mismatched or reordered columns
if list(X_train.columns) != list(X_test.columns):
    raise ValueError("X_train and X_test columns do not match.")

# ============================================================
# TARGET ENCODING
# ============================================================

# Convert string grade labels (e.g. "A", "B", "C") into integers (0, 1, 2, ...)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)   # fit on train labels only
y_test = label_encoder.transform(y_test_raw)          # reuse same mapping on test labels
num_classes = len(label_encoder.classes_)

print(f"Grade classes    : {list(label_encoder.classes_)}")

# ============================================================
# FEATURE TYPES
# ============================================================

# Split columns into numeric vs categorical so each type gets its own preprocessing
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()

# ============================================================
# PREPROCESSING
# Both SVM and Logistic Regression are distance/coefficient based,
# so numeric features are scaled (unlike the XGB+RF script, where
# tree models don't need scaling).
# ============================================================

preprocessor = ColumnTransformer(transformers=[
    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),   # fill missing numeric values with the median
            ("scaler", StandardScaler())                     # standardize numeric values (mean=0, std=1)
        ]),
        numeric_features
    ),
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),      # fill missing categories with the mode
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))  # one-hot encode categories
        ]),
        categorical_features
    )
])

print("\nPreprocessing...")
X_train_processed = preprocessor.fit_transform(X_train)  # fit preprocessing on train, then transform train
X_test_processed = preprocessor.transform(X_test)         # reuse the fitted preprocessing to transform test only

print(f"Processed shape  : train={X_train_processed.shape}, test={X_test_processed.shape}")


def evaluate(y_true, y_pred):
    # Compute a standard set of classification metrics (weighted across classes)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0)
    }


# ============================================================
# 1. SVM Baseline
# ============================================================

print("\n" + "-" * 55)
print("1. SVM Baseline")
print("-" * 55)

# Train a default/untuned SVM as a reference point to measure tuning improvement against
svm_baseline = SVC(C=1, kernel="rbf", gamma="scale", probability=True, random_state=42)
svm_baseline.fit(X_train_processed, y_train)

baseline_pred = svm_baseline.predict(X_test_processed)
baseline_result = evaluate(y_test, baseline_pred)

print(f"Accuracy : {baseline_result['Accuracy']:.4f}")
print(f"F1       : {baseline_result['F1']:.4f}")
print(f"Precision: {baseline_result['Precision']:.4f}")
print(f"Recall   : {baseline_result['Recall']:.4f}")


# ============================================================
# 2. Fine-tuning SVM
# ============================================================

print("\n" + "-" * 55)
print("2. Fine-tuning SVM")
print("-" * 55)

# Search space of SVM hyperparameters to try
param_grid = {
    "C": [0.1, 0.5, 1, 5, 10, 20, 50, 100],
    "kernel": ["rbf", "poly", "sigmoid"],
    "gamma": ["scale", "auto", 0.0001, 0.001, 0.01, 0.1],
    "degree": [2, 3, 4],
    "class_weight": [None, "balanced"]
}

# 3-fold stratified CV keeps class proportions balanced across folds during tuning
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Randomly sample 30 hyperparameter combinations and pick the best by weighted F1
search = RandomizedSearchCV(
    estimator=SVC(probability=True, random_state=42),
    param_distributions=param_grid,
    n_iter=30,
    scoring="f1_weighted",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

search.fit(X_train_processed, y_train)  # run the search on the full training set

svm_model = search.best_estimator_          # best SVM found, already refit on all training data
svm_pred = svm_model.predict(X_test_processed)
svm_prob = svm_model.predict_proba(X_test_processed)   # class probabilities, needed later for the hybrid blend
svm_result = evaluate(y_test, svm_pred)

print(f"Best CV F1      : {search.best_score_:.4f}")
print(f"Best parameters : {search.best_params_}")

print("\nSVM Tuning Results")
print("-" * 55)
print(f"{'Metric':<12}{'Before':>12}{'After':>12}{'Change':>12}")
print("-" * 48)

# Show how much tuning improved (or hurt) each metric vs. the baseline SVM
for metric in ["Accuracy", "F1", "Precision", "Recall"]:
    before = baseline_result[metric]
    after = svm_result[metric]
    print(f"{metric:<12}{before:>12.4f}{after:>12.4f}{after - before:>+12.4f}")


# ============================================================
# 3. Logistic Regression
# A linear, convex model with well-calibrated probabilities.
# It makes very different mistakes than an RBF-kernel SVM,
# which is what makes it a useful hybrid partner rather than
# just a weaker duplicate of the same decision boundary.
# ============================================================

print("\n" + "-" * 55)
print("3. Logistic Regression")
print("-" * 55)

# Train a logistic regression model as the second half of the hybrid
lr_model = LogisticRegression(
    max_iter=2000,                 # allow enough iterations to converge
    class_weight="balanced",       # compensate for any class imbalance
    random_state=42,
    n_jobs=-1
)

lr_model.fit(X_train_processed, y_train)

lr_pred = lr_model.predict(X_test_processed)
lr_prob = lr_model.predict_proba(X_test_processed)   # class probabilities, needed later for the hybrid blend
lr_result = evaluate(y_test, lr_pred)

print(f"Accuracy : {lr_result['Accuracy']:.4f}")
print(f"F1       : {lr_result['F1']:.4f}")
print(f"Precision: {lr_result['Precision']:.4f}")
print(f"Recall   : {lr_result['Recall']:.4f}")


# ============================================================
# 4. Selecting Hybrid Weight
# ============================================================

print("\n" + "-" * 55)
print("4. Selecting Hybrid Weight")
print("-" * 55)

# Separate 3-fold split used only to generate honest out-of-fold (OOF)
# probabilities on the TRAINING data, so the weight is chosen without
# ever peeking at the test set
weight_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

svm_oof_prob = np.zeros((len(X_train_processed), num_classes))
lr_oof_prob = np.zeros((len(X_train_processed), num_classes))

for fold, (train_idx, valid_idx) in enumerate(weight_cv.split(X_train_processed, y_train), start=1):

    # Clone = same hyperparameters, fresh untrained copy, so each fold trains independently
    svm_fold = clone(svm_model)
    svm_fold.fit(X_train_processed[train_idx], y_train[train_idx])
    svm_oof_prob[valid_idx] = svm_fold.predict_proba(X_train_processed[valid_idx])

    lr_fold = clone(lr_model)
    lr_fold.fit(X_train_processed[train_idx], y_train[train_idx])
    lr_oof_prob[valid_idx] = lr_fold.predict_proba(X_train_processed[valid_idx])

    print(f"Fold {fold}/3 completed")


best_weight = 0.5
best_f1 = -1

print("\nWeight Evaluation")
print("-" * 45)
print(f"{'SVM':>6}{'LR':>6}{'Accuracy':>12}{'F1':>10}")
print("-" * 45)

# Try several SVM/LR blend ratios and keep whichever gives the best OOF F1
for svm_weight in [0.5, 0.6, 0.7, 0.8, 0.9]:

    lr_weight = 1 - svm_weight
    hybrid_oof_prob = svm_weight * svm_oof_prob + lr_weight * lr_oof_prob   # weighted average of probabilities
    hybrid_oof_pred = hybrid_oof_prob.argmax(axis=1)                       # pick the highest-probability class
    hybrid_oof_result = evaluate(y_train, hybrid_oof_pred)

    print(
        f"{svm_weight:>6.1f}"
        f"{lr_weight:>6.1f}"
        f"{hybrid_oof_result['Accuracy']:>12.4f}"
        f"{hybrid_oof_result['F1']:>10.4f}"
    )

    if hybrid_oof_result["F1"] > best_f1:
        best_f1 = hybrid_oof_result["F1"]
        best_weight = svm_weight

lr_weight = 1 - best_weight

print(f"\nSelected weights: SVM={best_weight:.1f}, Logistic Regression={lr_weight:.1f}")
print(f"Best Hybrid CV F1: {best_f1:.4f}")


# ============================================================
# 5. Final Hybrid Model
# ============================================================

print("\n" + "-" * 55)
print("5. Final Hybrid Model")
print("-" * 55)

# Combine the two models' test-set probabilities using the chosen weights,
# then take the class with the highest blended probability as the final prediction
final_prob = best_weight * svm_prob + lr_weight * lr_prob
final_pred = final_prob.argmax(axis=1)
hybrid_result = evaluate(y_test, final_pred)

print("\nFinal Model Results")
print("-" * 68)
print(f"{'Model':<20}{'Accuracy':>12}{'F1':>12}{'Precision':>12}{'Recall':>12}")
print("-" * 68)

# Side-by-side comparison of SVM alone, LR alone, and the hybrid on the test set
for name, result in [
    ("SVM", svm_result),
    ("Logistic Regression", lr_result),
    ("SVM + LR Hybrid", hybrid_result)
]:
    print(
        f"{name:<20}"
        f"{result['Accuracy']:>12.4f}"
        f"{result['F1']:>12.4f}"
        f"{result['Precision']:>12.4f}"
        f"{result['Recall']:>12.4f}"
    )

print("-" * 68)
print(f"Hybrid weights: SVM={best_weight:.1f}, Logistic Regression={lr_weight:.1f}")

# Persist every trained component so the exact hybrid pipeline can be reloaded later
joblib.dump(svm_model, os.path.join(ARTIFACTS_DIR, "svm_grade_model.pkl"))
joblib.dump(preprocessor, os.path.join(ARTIFACTS_DIR, "svm_preprocessor.pkl"))
joblib.dump(label_encoder, os.path.join(ARTIFACTS_DIR, "svm_label_encoder.pkl"))
joblib.dump(feature_columns, os.path.join(ARTIFACTS_DIR, "svm_feature_columns.joblib"))
joblib.dump(lr_model, os.path.join(ARTIFACTS_DIR, "logistic_regression_grade_model.pkl"))
joblib.dump(
    {"svm_weight": best_weight, "lr_weight": lr_weight},
    os.path.join(ARTIFACTS_DIR, "svm_lr_hybrid_weights.joblib")
)

print("\n" + "-" * 55)
print("Artifacts saved successfully")
print("-" * 55)
print("svm_grade_model.pkl")
print("svm_preprocessor.pkl")
print("svm_label_encoder.pkl")
print("svm_feature_columns.joblib")
print("logistic_regression_grade_model.pkl")
print("svm_lr_hybrid_weights.joblib")
print("\nTraining complete.")