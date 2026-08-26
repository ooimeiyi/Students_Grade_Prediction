# ============================================================
# ANN GRADE PREDICTION
# ============================================================

import os
import joblib
import pandas as pd
import warnings

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold
)

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

ARTIFACTS_DIR = "artifacts"
TARGET = "Grade"

os.makedirs(
    ARTIFACTS_DIR,
    exist_ok=True
)


# ============================================================
# 1. LOAD PREPROCESSED TRAIN / TEST DATA
# ============================================================

X_train = pd.read_csv(
    os.path.join(
        ARTIFACTS_DIR,
        "X_train_raw.csv"
    )
)

X_test = pd.read_csv(
    os.path.join(
        ARTIFACTS_DIR,
        "X_test_raw.csv"
    )
)

y_train = pd.read_csv(
    os.path.join(
        ARTIFACTS_DIR,
        "y_train.csv"
    )
)[TARGET]

y_test = pd.read_csv(
    os.path.join(
        ARTIFACTS_DIR,
        "y_test.csv"
    )
)[TARGET]


# ============================================================
# OUTPUT
# ============================================================

print("============================================")
print("ANN STUDENT GRADE PREDICTION")
print("============================================")

print("\nLoaded preprocessing artifacts:")

print(" - artifacts/X_train_raw.csv")
print(" - artifacts/X_test_raw.csv")
print(" - artifacts/y_train.csv")
print(" - artifacts/y_test.csv")


print("\nTraining samples:")
print(len(X_train))

print("Testing samples :")
print(len(X_test))


# ============================================================
# 2. LOAD FEATURE COLUMNS
# ============================================================

feature_columns_path = os.path.join(
    ARTIFACTS_DIR,
    "feature_columns.joblib"
)

if os.path.exists(feature_columns_path):

    feature_columns = joblib.load(
        feature_columns_path
    )

    print("\nFeature columns loaded from:")
    print(" - artifacts/feature_columns.joblib")

    # Ensure the same feature order as preprocessing
    X_train = X_train[feature_columns]
    X_test = X_test[feature_columns]

else:

    feature_columns = X_train.columns.tolist()

    print(
        "\nWARNING: feature_columns.joblib not found."
    )

    print(
        "Using columns from X_train_raw.csv."
    )


# ============================================================
# 3. CHECK TARGET
# ============================================================

if TARGET in X_train.columns:

    raise ValueError(
        "ERROR: Grade target column is present "
        "inside X_train."
    )


if TARGET in X_test.columns:

    raise ValueError(
        "ERROR: Grade target column is present "
        "inside X_test."
    )


# ============================================================
# 4. CHECK TRAIN / TEST COLUMNS
# ============================================================

if list(X_train.columns) != list(X_test.columns):

    raise ValueError(
        "ERROR: Training and testing feature "
        "columns do not match."
    )


print("\nFeature count:")
print(len(X_train.columns))


# ============================================================
# 5. ENCODE TARGET
# ============================================================

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(
    y_train
)

y_test_encoded = label_encoder.transform(
    y_test
)


print("\nGrade classes:")
print(label_encoder.classes_)


# ============================================================
# 6. FEATURE TYPES
# ============================================================

numeric_features = X_train.select_dtypes(
    include=[
        "int64",
        "float64"
    ]
).columns.tolist()


categorical_features = X_train.select_dtypes(
    include=[
        "object",
        "category",
        "bool",
        "string"
    ]
).columns.tolist()


print("\nNumeric features:")

for column in numeric_features:
    print(" -", column)


print("\nCategorical features:")

for column in categorical_features:
    print(" -", column)


# ============================================================
# 7. PREPROCESSING
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
# 8. PREPROCESS TRAIN / TEST
# ============================================================

print("\n============================================")
print("MODEL-SPECIFIC PREPROCESSING")
print("============================================")


X_train_processed = preprocessor.fit_transform(
    X_train
)


X_test_processed = preprocessor.transform(
    X_test
)


print(
    "\nProcessed training shape:",
    X_train_processed.shape
)

print(
    "Processed testing shape :",
    X_test_processed.shape
)


# ============================================================
# 9. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    y_true,
    y_pred
):

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

print("\n============================================")
print("BASELINE ANN")
print("============================================")

# two hidden layers: 64 neurons in 1st and  32 neurons in 2nd layer

baseline_model = MLPClassifier(
    hidden_layer_sizes=(64, 32), 
    activation="relu", #activation function for the hidden layers
    solver="adam", # updates weights based on training data
    alpha=0.0001,
    batch_size=32,
    learning_rate_init=0.001,
    max_iter=500, # allows model up to iterate through the training data multiple times
    early_stopping=True, # stops training if validation score does not improve
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)

# train and predict using the baseline model
baseline_model.fit(
    X_train_processed,
    y_train_encoded
)


baseline_pred = baseline_model.predict(
    X_test_processed
)


before_results = evaluate_model(
    y_test_encoded,
    baseline_pred
)


print("\n============================================")
print("BEFORE FINE-TUNING")
print("============================================")


print(
    f"Accuracy : "
    f"{before_results['Accuracy']:.4f}"
)


print(
    f"F1 Score : "
    f"{before_results['F1']:.4f}"
)


print(
    f"Precision: "
    f"{before_results['Precision']:.4f}"
)


print(
    f"Recall   : "
    f"{before_results['Recall']:.4f}"
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

    y_train_encoded
)


# ============================================================
# 12. BEST MODEL
# ============================================================

print("\n============================================")
print("BEST ANN MODEL")
print("============================================")


print("\nBest Parameters:")

print(
    random_search.best_params_
)


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

    y_test_encoded,

    tuned_pred
)


print("\n============================================")
print("AFTER FINE-TUNING")
print("============================================")


print(
    f"Accuracy : "
    f"{after_results['Accuracy']:.4f}"
)


print(
    f"F1 Score : "
    f"{after_results['F1']:.4f}"
)


print(
    f"Precision: "
    f"{after_results['Precision']:.4f}"
)


print(
    f"Recall   : "
    f"{after_results['Recall']:.4f}"
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


print(
    "-" * 54
)


for metric in [

    "Accuracy",

    "F1",

    "Precision",

    "Recall"

]:

    before = before_results[
        metric
    ]

    after = after_results[
        metric
    ]

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


# ============================================================
# 16. SAVE MODEL FEATURES
# ============================================================

joblib.dump(

    feature_columns,

    os.path.join(

        ARTIFACTS_DIR,

        "ann_feature_columns.joblib"
    )
)


# ============================================================
# 17. FINAL OUTPUT
# ============================================================

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


print(
    "artifacts/ann_feature_columns.joblib"
)


print("\n============================================")
print("DATA SOURCE CHECK")
print("============================================")


print(
    "Training data:"
)

print(
    " - artifacts/X_train_raw.csv"
)


print(
    "Testing data:"
)

print(
    " - artifacts/X_test_raw.csv"
)


print(
    "Training target:"
)

print(
    " - artifacts/y_train.csv"
)


print(
    "Testing target:"
)

print(
    " - artifacts/y_test.csv"
)


print("\n============================================")
print("ANN TRAINING COMPLETE")
print("============================================")