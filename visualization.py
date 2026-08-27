import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


# ============================================================
# SETTINGS
# ============================================================

ARTIFACTS_DIR = "artifacts"

models = {
    "XGBoost": {
        "model": "xgboost_grade_model.pkl",
        "preprocessor": "xgboost_preprocessor.pkl",
        "encoder": "xgboost_label_encoder.pkl",
        "cmap": "Blues"
    },

    "ANN": {
        "model": "ann_grade_model.pkl",
        "preprocessor": "ann_preprocessor.pkl",
        "encoder": "ann_label_encoder.pkl",
        "cmap": "Greens"
    },

    "SVM": {
        "model": "svm_grade_model.pkl",
        "preprocessor": "svm_preprocessor.pkl",
        "encoder": "svm_label_encoder.pkl",
        "cmap": "YlOrBr"
    }
}


# ============================================================
# LOAD TEST DATA
# ============================================================

X_test = pd.read_csv(
    os.path.join(
        ARTIFACTS_DIR,
        "X_test_raw.csv"
    )
)

y_test = pd.read_csv(
    os.path.join(
        ARTIFACTS_DIR,
        "y_test.csv"
    )
)["Grade"]


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

OUTPUT_DIR = "confusion_matrices"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# PROCESS EACH MODEL
# ============================================================

for model_name, files in models.items():

    print("\n============================================")
    print(model_name)
    print("============================================")


    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = joblib.load(
        os.path.join(
            ARTIFACTS_DIR,
            files["model"]
        )
    )


    # ========================================================
    # LOAD PREPROCESSOR
    # ========================================================

    preprocessor = joblib.load(
        os.path.join(
            ARTIFACTS_DIR,
            files["preprocessor"]
        )
    )


    # ========================================================
    # LOAD LABEL ENCODER
    # ========================================================

    label_encoder = joblib.load(
        os.path.join(
            ARTIFACTS_DIR,
            files["encoder"]
        )
    )


    # ========================================================
    # PREPROCESS TEST DATA
    # ========================================================

    X_test_processed = preprocessor.transform(
        X_test
    )


    # ========================================================
    # PREDICT
    # ========================================================

    prediction_encoded = model.predict(
        X_test_processed
    )

    predicted_labels = label_encoder.inverse_transform(
        prediction_encoded
    )


    # ========================================================
    # TRUE LABELS
    # ========================================================

    true_labels = y_test


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    labels = label_encoder.classes_

    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=labels
    )


    # ========================================================
    # PRINT CONFUSION MATRIX
    # ========================================================

    print("\nTrue Label vs Predicted Label")

    print(cm)


    # ========================================================
    # DRAW GRAPH
    # ========================================================

    plt.figure(
        figsize=(8, 6)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=files["cmap"],
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="white"
    )


    # ========================================================
    # TITLES
    # ========================================================

    plt.title(
        f"{model_name} Confusion Matrix",
        fontsize=15,
        fontweight="bold"
    )

    plt.xlabel(
        "Predicted Label",
        fontsize=12
    )

    plt.ylabel(
        "True Label",
        fontsize=12
    )


    # ========================================================
    # SAVE GRAPH
    # ========================================================

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{model_name.lower()}_confusion_matrix.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )


    # ========================================================
    # SHOW GRAPH
    # ========================================================

    plt.show()

    plt.close()


    print(
        f"\nGraph saved: {output_file}"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n============================================")
print("ALL CONFUSION MATRICES CREATED")
print("============================================")

print("\nFiles:")

print(
    "confusion_matrices/xgboost_confusion_matrix.png"
)

print(
    "confusion_matrices/ann_confusion_matrix.png"
)

print(
    "confusion_matrices/svm_confusion_matrix.png"
)