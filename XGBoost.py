# ============================================================
# XGBOOST GRADE PREDICTION
# ============================================================

import os
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from xgboost import XGBClassifier

# ============================================================
# SETTINGS
# ============================================================

FILE_NAME = "Students_Performance_Dataset_Clean.csv"
ARTIFACTS_DIR = "artifacts"
TARGET = "Grade"

os.makedirs(
    ARTIFACTS_DIR,
    exist_ok=True
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(FILE_NAME)

# ============================================================
# 2. REMOVE UNNECESSARY COLUMNS
# ============================================================

remove_columns = [
    "Student_ID",
    "StudentID",
    "ID",
    "First_Name",
    "Last_Name",
    "Email",
    "Total_Score"
]

remove_columns = [
    col for col in remove_columns
    if col in df.columns
]

df = df.drop(
    columns=remove_columns
)

# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET]

# ============================================================
# 4. ENCODE TARGET
# ============================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

num_classes = len(
    label_encoder.classes_
)

# ============================================================
# OUTPUT
# ============================================================

print("============================================")
print("XGBOOST STUDENT GRADE PREDICTION")
print("============================================")

print("\nGrade classes:")
print(label_encoder.classes_)

# ============================================================
# 5. FEATURE TYPES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=[
        "object",
        "category",
        "bool",
        "string"
    ]
).columns.tolist()

# ============================================================
# 6. PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples :",
    len(X_test)
)

# ============================================================
# 8. PREPROCESS
# ============================================================

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)

# ============================================================
# 9. EVALUATION FUNCTION
# ============================================================

def evaluate_model(y_true, y_pred):

    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "F1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "Precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    }

# ============================================================
# 10. BEFORE FINE-TUNING
# ============================================================

baseline_model = XGBClassifier(
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

baseline_model.fit(
    X_train_processed,
    y_train
)

baseline_pred = baseline_model.predict(
    X_test_processed
)

before_results = evaluate_model(
    y_test,
    baseline_pred
)

print("\n============================================")
print("BEFORE FINE-TUNING")
print("============================================")

print(
    f"Accuracy : {before_results['Accuracy']:.4f}"
)

print(
    f"F1 Score : {before_results['F1']:.4f}"
)

print(
    f"Precision: {before_results['Precision']:.4f}"
)

print(
    f"Recall : {before_results['Recall']:.4f}"
)

# ============================================================
# 11. FINE-TUNING
# ============================================================

print("\n============================================")
print("FINE-TUNING XGBOOST")
print("============================================")

param_grid = {
    "n_estimators": [
        100,
        200,
        300,
        400,
        500
    ],

    "max_depth": [
        2,
        3,
        4,
        5,
        6
    ],

    "learning_rate": [
        0.01,
        0.03,
        0.05,
        0.08,
        0.10,
        0.15
    ],

    "min_child_weight": [
        1,
        3,
        5,
        7,
        10
    ],

    "gamma": [
        0,
        0.1,
        0.3,
        0.5,
        1
    ],

    "subsample": [
        0.7,
        0.8,
        0.9,
        1.0
    ],

    "colsample_bytree": [
        0.7,
        0.8,
        0.9,
        1.0
    ],

    "reg_alpha": [
        0,
        0.01,
        0.1,
        1
    ],

    "reg_lambda": [
        1,
        2,
        5,
        10,
        20
    ]
}

tuning_model = XGBClassifier(
    objective="multi:softprob",
    num_class=num_classes,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

random_search = RandomizedSearchCV(
    estimator=tuning_model,
    param_distributions=param_grid,
    n_iter=50,
    scoring="f1_weighted",
    cv=cv,
    verbose=0,
    random_state=42,
    n_jobs=-1
)

random_search.fit(
    X_train_processed,
    y_train
)

# ============================================================
# 12. BEST MODEL
# ============================================================

print("\nBest Parameters:")
print(random_search.best_params_)

print(
    f"\nBest CV F1 Score: "
    f"{random_search.best_score_:.4f}"
)

# ============================================================
# 13. AFTER FINE-TUNING
# ============================================================

best_model = random_search.best_estimator_

tuned_pred = best_model.predict(
    X_test_processed
)

after_results = evaluate_model(
    y_test,
    tuned_pred
)

print("\n============================================")
print("AFTER FINE-TUNING")
print("============================================")

print(
    f"Accuracy : {after_results['Accuracy']:.4f}"
)

print(
    f"F1 Score : {after_results['F1']:.4f}"
)

print(
    f"Precision: {after_results['Precision']:.4f}"
)

print(
    f"Recall : {after_results['Recall']:.4f}"
)

# ============================================================
# 14. BEFORE VS AFTER
# ============================================================

print("\n============================================")
print("BEFORE vs AFTER FINE-TUNING")
print("============================================")

print(
    f"{'Metric':<15}"
    f"{'Before':>12}"
    f"{'After':>12}"
    f"{'Improvement':>15}"
)

print("-" * 54)

for metric in [
    "Accuracy",
    "F1",
    "Precision",
    "Recall"
]:

    before = before_results[metric]
    after = after_results[metric]

    improvement = after - before

    print(
        f"{metric:<15}"
        f"{before:>12.4f}"
        f"{after:>12.4f}"
        f"{improvement:>15.4f}"
    )

# ============================================================
# 15. SAVE ARTIFACTS
# ============================================================

joblib.dump(
    best_model,
    os.path.join(
        ARTIFACTS_DIR,
        "xgboost_grade_model.pkl"
    )
)

joblib.dump(
    preprocessor,
    os.path.join(
        ARTIFACTS_DIR,
        "xgboost_preprocessor.pkl"
    )
)

joblib.dump(
    label_encoder,
    os.path.join(
        ARTIFACTS_DIR,
        "xgboost_label_encoder.pkl"
    )
)

print("\n============================================")
print("XGBOOST ARTIFACTS SAVED")
print("============================================")

print(
    "artifacts/xgboost_grade_model.pkl"
)

print(
    "artifacts/xgboost_preprocessor.pkl"
)

print(
    "artifacts/xgboost_label_encoder.pkl"
)