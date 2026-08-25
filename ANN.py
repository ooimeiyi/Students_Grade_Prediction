# ============================================================
# ANN GRADE PREDICTION
# ============================================================

import os
import joblib
import pandas as pd
import warnings

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

warnings.filterwarnings("ignore")

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

# ============================================================
# OUTPUT
# ============================================================

print("============================================")
print("ANN STUDENT GRADE PREDICTION")
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
        ),
        (
            "scaler",
            StandardScaler()
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
print("FINE-TUNING ANN")
print("============================================")

param_grid = {
    "hidden_layer_sizes": [
        (32,),
        (64,),
        (128,),
        (64, 32),
        (128, 64),
        (128, 64, 32)
    ],

    "activation": [
        "relu",
        "tanh"
    ],

    "alpha": [
        0.00001,
        0.0001,
        0.001,
        0.01
    ],

    "learning_rate_init": [
        0.0001,
        0.0005,
        0.001,
        0.005,
        0.01
    ],

    "batch_size": [
        16,
        32,
        64,
        128
    ]
}

tuning_model = MLPClassifier(
    solver="adam",
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

random_search = RandomizedSearchCV(
    estimator=tuning_model,
    param_distributions=param_grid,
    n_iter=30,
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
        "ann_grade_model.pkl"
    )
)

joblib.dump(
    preprocessor,
    os.path.join(
        ARTIFACTS_DIR,
        "ann_preprocessor.pkl"
    )
)

joblib.dump(
    label_encoder,
    os.path.join(
        ARTIFACTS_DIR,
        "ann_label_encoder.pkl"
    )
)

print("\n============================================")
print("ANN ARTIFACTS SAVED")
print("============================================")

print(
    "artifacts/ann_grade_model.pkl"
)

print(
    "artifacts/ann_preprocessor.pkl"
)

print(
    "artifacts/ann_label_encoder.pkl"
)