# SVM GRADE PREDICTION

import os
import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# Settings
ARTIFACTS_DIR = "artifacts"
TARGET = "Grade"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# Load data
X_train = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_train_raw.csv"))
X_test = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_test_raw.csv"))
y_train = pd.read_csv(os.path.join(ARTIFACTS_DIR, "y_train.csv"))[TARGET]
y_test = pd.read_csv(os.path.join(ARTIFACTS_DIR, "y_test.csv"))[TARGET]


# Load feature columns
feature_columns_path = os.path.join(ARTIFACTS_DIR, "feature_columns.joblib")

if os.path.exists(feature_columns_path):
    feature_columns = joblib.load(feature_columns_path)
    X_train = X_train[feature_columns]
    X_test = X_test[feature_columns]
else:
    feature_columns = X_train.columns.tolist()


# Validate data
if TARGET in X_train.columns:
    raise ValueError("ERROR: Grade target column is present inside X_train.")

if TARGET in X_test.columns:
    raise ValueError("ERROR: Grade target column is present inside X_test.")

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("ERROR: Training and testing feature columns do not match.")


# Encode target
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


# Identify feature types
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()


# Preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# Preprocess data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# Evaluation function
def evaluate_model(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0)
    }


# Before fine-tuning
baseline_model = SVC(C=1, kernel="rbf", gamma="scale")
baseline_model.fit(X_train_processed, y_train_encoded)
baseline_pred = baseline_model.predict(X_test_processed)
before_results = evaluate_model(y_test_encoded, baseline_pred)


# Fine-tuning
param_grid = {
    "C": [0.1, 0.5, 1, 5, 10, 20, 50, 100],
    "kernel": ["rbf", "poly", "sigmoid"],
    "gamma": ["scale", "auto", 0.0001, 0.001, 0.01, 0.1],
    "degree": [2, 3, 4],
    "class_weight": [None, "balanced"]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=SVC(),
    param_distributions=param_grid,
    n_iter=30,
    scoring="f1_weighted",
    cv=cv,
    verbose=0,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_processed, y_train_encoded)


# After fine-tuning
best_model = random_search.best_estimator_
tuned_pred = best_model.predict(X_test_processed)
after_results = evaluate_model(y_test_encoded, tuned_pred)


# Results
print("\nSVM RESULTS")

print("\nBefore Fine-Tuning")
print(f"Accuracy : {before_results['Accuracy']:.4f}")
print(f"F1 Score : {before_results['F1']:.4f}")
print(f"Precision: {before_results['Precision']:.4f}")
print(f"Recall   : {before_results['Recall']:.4f}")

print("\nAfter Fine-Tuning")
print(f"Accuracy : {after_results['Accuracy']:.4f}")
print(f"F1 Score : {after_results['F1']:.4f}")
print(f"Precision: {after_results['Precision']:.4f}")
print(f"Recall   : {after_results['Recall']:.4f}")


# Before vs After
print("\nBefore vs After Fine-Tuning")
print(f"{'Metric':<12}{'Before':>12}{'After':>12}{'Change':>12}")
print("-" * 48)

for metric in ["Accuracy", "F1", "Precision", "Recall"]:
    before = before_results[metric]
    after = after_results[metric]
    change = after - before
    print(f"{metric:<12}{before:>12.4f}{after:>12.4f}{change:>12.4f}")


# Final result
print("\nFinal Result")
print("Best Parameters:", random_search.best_params_)
print(f"Accuracy : {after_results['Accuracy']:.4f}")
print(f"F1 Score : {after_results['F1']:.4f}")
print(f"Precision: {after_results['Precision']:.4f}")
print(f"Recall   : {after_results['Recall']:.4f}")


# Save artifacts
joblib.dump(best_model, os.path.join(ARTIFACTS_DIR, "svm_grade_model.pkl"))
joblib.dump(preprocessor, os.path.join(ARTIFACTS_DIR, "svm_preprocessor.pkl"))
joblib.dump(label_encoder, os.path.join(ARTIFACTS_DIR, "svm_label_encoder.pkl"))

print("\nSVM training complete.")