import os
import joblib
import pandas as pd
import warnings
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.base import clone
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


ARTIFACTS_DIR = "artifacts"
TARGET = "Grade"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


print("\n" + "=" * 55)
print("XGBoost + Random Forest Grade Prediction")
print("=" * 55)


X_train = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_train_raw.csv"))
X_test = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_test_raw.csv"))
y_train_raw = pd.read_csv(os.path.join(ARTIFACTS_DIR, "y_train.csv"))[TARGET]
y_test_raw = pd.read_csv(os.path.join(ARTIFACTS_DIR, "y_test.csv"))[TARGET]

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")


feature_path = os.path.join(ARTIFACTS_DIR, "feature_columns.joblib")

if os.path.exists(feature_path):
    feature_columns = joblib.load(feature_path)
    X_train = X_train[feature_columns]
    X_test = X_test[feature_columns]
else:
    feature_columns = X_train.columns.tolist()


if TARGET in X_train.columns:
    raise ValueError("Grade exists inside X_train.")

if TARGET in X_test.columns:
    raise ValueError("Grade exists inside X_test.")

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("X_train and X_test columns do not match.")


label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)
y_test = label_encoder.transform(y_test_raw)
num_classes = len(label_encoder.classes_)

print(f"Grade classes    : {list(label_encoder.classes_)}")


numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()


preprocessor = ColumnTransformer(transformers=[
    (
        "num",
        Pipeline([("imputer", SimpleImputer(strategy="median"))]),
        numeric_features
    ),
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]),
        categorical_features
    )
])


print("\nPreprocessing...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"Processed shape  : train={X_train_processed.shape}, test={X_test_processed.shape}")


def evaluate(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0)
    }


print("\n" + "-" * 55)
print("1. XGBoost Baseline")
print("-" * 55)

xgb_baseline = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0,
    reg_lambda=1,
    objective="multi:softprob",
    num_class=num_classes,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

xgb_baseline.fit(X_train_processed, y_train)

baseline_pred = xgb_baseline.predict(X_test_processed)
baseline_result = evaluate(y_test, baseline_pred)

print(f"Accuracy : {baseline_result['Accuracy']:.4f}")
print(f"F1       : {baseline_result['F1']:.4f}")
print(f"Precision: {baseline_result['Precision']:.4f}")
print(f"Recall   : {baseline_result['Recall']:.4f}")


print("\n" + "-" * 55)
print("2. Fine-tuning XGBoost")
print("-" * 55)

param_grid = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.10, 0.15],
    "min_child_weight": [1, 3, 5, 7, 10],
    "gamma": [0, 0.1, 0.3, 0.5, 1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0, 0.01, 0.1, 1],
    "reg_lambda": [1, 2, 5, 10, 20]
}

xgb_tuning = XGBClassifier(
    objective="multi:softprob",
    num_class=num_classes,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=xgb_tuning,
    param_distributions=param_grid,
    n_iter=50,
    scoring="accuracy",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

search.fit(X_train_processed, y_train)

xgb_model = search.best_estimator_
xgb_pred = xgb_model.predict(X_test_processed)
xgb_prob = xgb_model.predict_proba(X_test_processed)
xgb_result = evaluate(y_test, xgb_pred)

print(f"Best CV Accuracy : {search.best_score_:.4f}")
print(f"Best parameters  : {search.best_params_}")


print("\nXGBoost Tuning Results")
print("-" * 55)
print(f"{'Metric':<12}{'Before':>12}{'After':>12}{'Change':>12}")
print("-" * 48)

for metric in ["Accuracy", "F1", "Precision", "Recall"]:
    before = baseline_result[metric]
    after = xgb_result[metric]
    print(f"{metric:<12}{before:>12.4f}{after:>12.4f}{after - before:>+12.4f}")


print("\n" + "-" * 55)
print("3. Random Forest")
print("-" * 55)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_processed, y_train)

rf_pred = rf_model.predict(X_test_processed)
rf_prob = rf_model.predict_proba(X_test_processed)
rf_result = evaluate(y_test, rf_pred)

print(f"Accuracy : {rf_result['Accuracy']:.4f}")
print(f"F1       : {rf_result['F1']:.4f}")
print(f"Precision: {rf_result['Precision']:.4f}")
print(f"Recall   : {rf_result['Recall']:.4f}")


print("\n" + "-" * 55)
print("4. Selecting Hybrid Weight")
print("-" * 55)

weight_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

xgb_oof_prob = np.zeros((len(X_train_processed), num_classes))
rf_oof_prob = np.zeros((len(X_train_processed), num_classes))

for fold, (train_idx, valid_idx) in enumerate(weight_cv.split(X_train_processed, y_train), start=1):

    xgb_fold = clone(xgb_model)
    xgb_fold.fit(X_train_processed[train_idx], y_train[train_idx])
    xgb_oof_prob[valid_idx] = xgb_fold.predict_proba(X_train_processed[valid_idx])

    rf_fold = clone(rf_model)
    rf_fold.fit(X_train_processed[train_idx], y_train[train_idx])
    rf_oof_prob[valid_idx] = rf_fold.predict_proba(X_train_processed[valid_idx])

    print(f"Fold {fold}/3 completed")


best_weight = 0.5
best_accurancy = -1

print("\nWeight Evaluation")
print("-" * 45)
print(f"{'XGB':>6}{'RF':>6}{'Accuracy':>12}{'F1':>10}")
print("-" * 45)

for xgb_weight in [0.5, 0.6, 0.7, 0.8, 0.9]:

    rf_weight = 1 - xgb_weight
    hybrid_oof_prob = xgb_weight * xgb_oof_prob + rf_weight * rf_oof_prob
    hybrid_oof_pred = hybrid_oof_prob.argmax(axis=1)
    hybrid_oof_result = evaluate(y_train, hybrid_oof_pred)

    print(
        f"{xgb_weight:>6.1f}"
        f"{rf_weight:>6.1f}"
        f"{hybrid_oof_result['Accuracy']:>12.4f}"
        f"{hybrid_oof_result['F1']:>10.4f}"
    )

    if hybrid_oof_result["Accuracy"] > best_accurancy:
        best_accurancy = hybrid_oof_result["Accuracy"]
        best_weight = xgb_weight

rf_weight = 1 - best_weight

print(f"\nSelected weights: XGBoost={best_weight:.1f}, Random Forest={rf_weight:.1f}")
print(f"Best Hybrid CV Accuracy: {best_accurancy:.4f}")


print("\n" + "-" * 55)
print("5. Final Hybrid Model")
print("-" * 55)

final_prob = best_weight * xgb_prob + rf_weight * rf_prob
final_pred = final_prob.argmax(axis=1)
hybrid_result = evaluate(y_test, final_pred)


print("\nFinal Model Results")
print("-" * 68)
print(f"{'Model':<20}{'Accuracy':>12}{'F1':>12}{'Precision':>12}{'Recall':>12}")
print("-" * 68)

for name, result in [
    ("XGBoost", xgb_result),
    ("Random Forest", rf_result),
    ("XGB + RF Hybrid", hybrid_result)
]:
    print(
        f"{name:<20}"
        f"{result['Accuracy']:>12.4f}"
        f"{result['F1']:>12.4f}"
        f"{result['Precision']:>12.4f}"
        f"{result['Recall']:>12.4f}"
    )

print("-" * 68)
print(f"Hybrid weights: XGBoost={best_weight:.1f}, Random Forest={rf_weight:.1f}")


joblib.dump(xgb_model, os.path.join(ARTIFACTS_DIR, "xgboost_grade_model.pkl"))
joblib.dump(preprocessor, os.path.join(ARTIFACTS_DIR, "xgboost_preprocessor.pkl"))
joblib.dump(label_encoder, os.path.join(ARTIFACTS_DIR, "xgboost_label_encoder.pkl"))
joblib.dump(feature_columns, os.path.join(ARTIFACTS_DIR, "xgboost_feature_columns.joblib"))
joblib.dump(rf_model, os.path.join(ARTIFACTS_DIR, "random_forest_grade_model.pkl"))
joblib.dump({"xgb_weight": best_weight, "rf_weight": rf_weight}, os.path.join(ARTIFACTS_DIR, "hybrid_weights.joblib"))


print("\n" + "-" * 55)
print("Artifacts saved successfully")
print("-" * 55)
print("xgboost_grade_model.pkl")
print("xgboost_preprocessor.pkl")
print("xgboost_label_encoder.pkl")
print("xgboost_feature_columns.joblib")
print("random_forest_grade_model.pkl")
print("hybrid_weights.joblib")
print("\nTraining complete.")