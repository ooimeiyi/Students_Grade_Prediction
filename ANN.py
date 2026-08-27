# ============================================================
# ANN GRADE PREDICTION
# ============================================================

import os
import joblib
import pandas as pd
import warnings

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")

# ============================================================
# SETTINGS
# ============================================================

ARTIFACTS_DIR = "artifacts"
TARGET = "Grade"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("============================================")
print("ANN STUDENT GRADE PREDICTION")
print("============================================")

X_train = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_train_raw.csv"))
X_test = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_test_raw.csv"))
y_train = pd.read_csv(os.path.join(ARTIFACTS_DIR, "y_train.csv"))[TARGET]
y_test = pd.read_csv(os.path.join(ARTIFACTS_DIR, "y_test.csv"))[TARGET]

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

# ============================================================
# LOAD FEATURE COLUMNS
# ============================================================

feature_columns_path = os.path.join(ARTIFACTS_DIR, "feature_columns.joblib")

if os.path.exists(feature_columns_path):
    feature_columns = joblib.load(feature_columns_path)
    X_train = X_train[feature_columns]
    X_test = X_test[feature_columns]
    print("\nFeature columns loaded successfully.")
else:
    feature_columns = X_train.columns.tolist()
    print("\nWARNING: feature_columns.joblib not found.")

# ============================================================
# VALIDATION
# ============================================================

if TARGET in X_train.columns:
    raise ValueError("ERROR: Grade exists inside X_train.")

if TARGET in X_test.columns:
    raise ValueError("ERROR: Grade exists inside X_test.")

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("ERROR: X_train and X_test columns do not match.")

print("\nNumber of features:", len(X_train.columns))

# ============================================================
# TARGET ENCODING
# ============================================================

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print("\nGrade classes:")
print(label_encoder.classes_)

# ============================================================
# FEATURE TYPES
# ============================================================

numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()

print("\nNumeric features:")
for col in numeric_features:
    print(" -", col)

print("\nCategorical features:")
for col in categorical_features:
    print(" -", col)

# ============================================================
# PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ============================================================
# PREPROCESS DATA
# ============================================================

print("\nPreprocessing data...")

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape :", X_test_processed.shape)

# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0)
    }

# ============================================================
# BASELINE ANN
# ============================================================

print("\n============================================")
print("BASELINE ANN")
print("============================================")

baseline_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size=32,
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)

baseline_model.fit(X_train_processed, y_train_encoded)
baseline_pred = baseline_model.predict(X_test_processed)
before_results = evaluate_model(y_test_encoded, baseline_pred)

print("\nBEFORE FINE-TUNING")
for metric, value in before_results.items():
    print(f"{metric:<10}: {value:.4f}")

# ============================================================
# FINE-TUNING ANN
# ============================================================

print("\n============================================")
print("FINE-TUNING ANN")
print("============================================")

param_grid = {
    "hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
    "activation": ["relu", "tanh"],
    "alpha": [0.00001, 0.0001, 0.001, 0.01],
    "learning_rate_init": [0.0001, 0.0005, 0.001, 0.005, 0.01],
    "batch_size": [16, 32, 64, 128]
}

tuning_model = MLPClassifier(
    solver="adam",
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=tuning_model,
    param_distributions=param_grid,
    n_iter=30,
    scoring="f1_weighted",
    cv=cv,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_processed, y_train_encoded)

# ============================================================
# BEST MODEL
# ============================================================

best_model = random_search.best_estimator_

print("\n============================================")
print("BEST ANN MODEL")
print("============================================")

print("\nBest Parameters:")
print(random_search.best_params_)
print(f"\nBest CV F1: {random_search.best_score_:.4f}")

# ============================================================
# AFTER FINE-TUNING
# ============================================================

tuned_pred = best_model.predict(X_test_processed)
after_results = evaluate_model(y_test_encoded, tuned_pred)

print("\nAFTER FINE-TUNING")
for metric, value in after_results.items():
    print(f"{metric:<10}: {value:.4f}")

# ============================================================
# BEFORE VS AFTER
# ============================================================

print("\n============================================")
print("BEFORE vs AFTER")
print("============================================")

print(f"{'Metric':<12}{'Before':>12}{'After':>12}{'Change':>12}")
print("-" * 48)

for metric in ["Accuracy", "F1", "Precision", "Recall"]:
    before = before_results[metric]
    after = after_results[metric]
    print(f"{metric:<12}{before:>12.4f}{after:>12.4f}{after - before:>+12.4f}")

# ============================================================
# SAVE ARTIFACTS
# ============================================================

joblib.dump(best_model, os.path.join(ARTIFACTS_DIR, "ann_grade_model.pkl"))
joblib.dump(preprocessor, os.path.join(ARTIFACTS_DIR, "ann_preprocessor.pkl"))
joblib.dump(label_encoder, os.path.join(ARTIFACTS_DIR, "ann_label_encoder.pkl"))
joblib.dump(feature_columns, os.path.join(ARTIFACTS_DIR, "ann_feature_columns.joblib"))

print("\n============================================")
print("ANN ARTIFACTS SAVED")
print("============================================")
print("ann_grade_model.pkl")
print("ann_preprocessor.pkl")
print("ann_label_encoder.pkl")
print("ann_feature_columns.joblib")
print("\nANN training complete.")