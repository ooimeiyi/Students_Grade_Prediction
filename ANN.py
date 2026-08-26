# ============================================================
# ANN GRADE PREDICTION
# ============================================================

import os
import joblib
import pandas as pd

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
# 1. LOAD TRAIN / TEST DATA
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

    X_train = X_train[
        feature_columns
    ]

    X_test = X_test[
        feature_columns
    ]

else:

    feature_columns = (
        X_train.columns.tolist()
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

if list(
    X_train.columns
) != list(
    X_test.columns
):

    raise ValueError(
        "ERROR: Training and testing feature "
        "columns do not match."
    )


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


# ============================================================
# 6. FEATURE TYPES
# ============================================================

numeric_features = (
    X_train
    .select_dtypes(
        include=[
            "int64",
            "float64"
        ]
    )
    .columns
    .tolist()
)


categorical_features = (
    X_train
    .select_dtypes(
        include=[
            "object",
            "category",
            "bool",
            "string"
        ]
    )
    .columns
    .tolist()
)


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

X_train_processed = (
    preprocessor.fit_transform(
        X_train
    )
)

X_test_processed = (
    preprocessor.transform(
        X_test
    )
)


# ============================================================
# 9. TRAIN ANN
# ============================================================

ann_model = MLPClassifier(

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


ann_model.fit(

    X_train_processed,

    y_train_encoded
)


# ============================================================
# 10. PREDICTION
# ============================================================

ann_pred = ann_model.predict(
    X_test_processed
)


# ============================================================
# 11. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test_encoded,
    ann_pred
)

f1 = f1_score(
    y_test_encoded,
    ann_pred,
    average="weighted",
    zero_division=0
)

precision = precision_score(
    y_test_encoded,
    ann_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test_encoded,
    ann_pred,
    average="weighted",
    zero_division=0
)


print("\n============================================")
print("ANN RESULTS")
print("============================================")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)


# ============================================================
# 12. SAVE ARTIFACTS
# ============================================================

joblib.dump(

    ann_model,

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


joblib.dump(

    feature_columns,

    os.path.join(

        ARTIFACTS_DIR,

        "ann_feature_columns.joblib"
    )
)


# ============================================================
# 13. COMPLETE
# ============================================================

print("\nANN training complete.")