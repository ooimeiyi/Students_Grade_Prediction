# ============================================================
# SVM + LOGISTIC REGRESSION HYBRID GRADE PREDICTION
# ============================================================

import os                      # file paths / creating the artifacts folder
import joblib                  # saving and loading trained models
import pandas as pd            # loading the CSV datasets
import warnings                # used to silence non-critical sklearn warnings
import numpy as np             # array math for the probability blending

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.base import clone   # makes an untrained copy of a fitted model, used for the CV loop

warnings.filterwarnings("ignore")  # hide sklearn's convergence/deprecation warnings from the console


# ============================================================
# HYBRID MODEL WRAPPER
# Combines the fitted SVM and Logistic Regression models into a
# single object with .predict() / .predict_proba(), so it can be
# saved and reloaded as one artifact instead of three separate pieces.
#
# NOTE: because this class is defined inside this script (rather
# than its own importable module), joblib will only be able to
# reload the saved .pkl correctly from THIS script. Loading it
# from a different script (e.g. app.py) would fail, since pickle
# needs to re-import the exact class definition it was saved with.
# If you ever want to load svm_lr_hybrid_model.pkl elsewhere, move
# this class into its own hybrid_model.py file and import it in
# both places instead.
# ============================================================

class SVMLogisticHybrid:

    def __init__(self, svm_model, lr_model, svm_weight, lr_weight):
        self.svm_model = svm_model          # fitted SVC (probability=True)
        self.lr_model = lr_model            # fitted LogisticRegression
        self.svm_weight = svm_weight        # blend weight for the SVM, e.g. 0.6
        self.lr_weight = lr_weight          # blend weight for LR, e.g. 0.4
        self.classes_ = svm_model.classes_  # encoded class labels (0..num_classes-1)

    def predict_proba(self, X):
        # Blend both models' class probabilities using the stored weights
        svm_prob = self.svm_model.predict_proba(X)
        lr_prob = self.lr_model.predict_proba(X)
        return self.svm_weight * svm_prob + self.lr_weight * lr_prob

    def predict(self, X):
        # Pick the class with the highest blended probability
        blended_prob = self.predict_proba(X)
        return self.classes_[np.argmax(blended_prob, axis=1)]

# ============================================================
# SETTINGS
# ============================================================

ARTIFACTS_DIR = "artifacts"     # folder where all inputs/outputs for this script live
TARGET = "Grade"                # name of the label column in y_train.csv / y_test.csv
os.makedirs(ARTIFACTS_DIR, exist_ok=True)   # create the folder if it doesn't exist yet

print("\n" + "=" * 55)
print("SVM + Logistic Regression Hybrid Grade Prediction")
print("=" * 55)

# ============================================================
# LOAD DATA
# ============================================================

# Read the already-split train/test features and labels produced by preprocessing.py
X_train = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_train_raw.csv"))
X_test = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_test_raw.csv"))
y_train_raw = pd.read_csv(os.path.join(ARTIFACTS_DIR, "y_train.csv"))[TARGET]
y_test_raw = pd.read_csv(os.path.join(ARTIFACTS_DIR, "y_test.csv"))[TARGET]

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# ============================================================
# FEATURE COLUMNS
# ============================================================

# If an earlier script already picked a fixed set of feature columns, reuse it
# so every model in the project trains on exactly the same inputs.
feature_path = os.path.join(ARTIFACTS_DIR, "feature_columns.joblib")

if os.path.exists(feature_path):
    feature_columns = joblib.load(feature_path)
    X_train = X_train[feature_columns]
    X_test = X_test[feature_columns]
else:
    feature_columns = X_train.columns.tolist()   # fall back to all available columns

# ============================================================
# VALIDATION
# ============================================================

# Guard rails: catch accidental leakage or a broken split before training starts.
if TARGET in X_train.columns:
    raise ValueError("Grade exists inside X_train.")

if TARGET in X_test.columns:
    raise ValueError("Grade exists inside X_test.")

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("X_train and X_test columns do not match.")

# ============================================================
# TARGET ENCODING
# ============================================================

# Convert text grades ("A", "B", ...) into integers (0, 1, ...) that sklearn models require.
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)   # learn the mapping from the training labels only
y_test = label_encoder.transform(y_test_raw)          # apply the same mapping to the test labels
num_classes = len(label_encoder.classes_)

print(f"Grade classes    : {list(label_encoder.classes_)}")

# ============================================================
# FEATURE TYPES
# ============================================================

# Split columns into numeric vs categorical so each type gets the right preprocessing.
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
            ("scaler", StandardScaler())                       # standardize numeric features to mean 0 / std 1
        ]),
        numeric_features
    ),
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),           # fill missing categories with the mode
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))  # one-hot encode categories
        ]),
        categorical_features
    )
])

print("\nPreprocessing...")
X_train_processed = preprocessor.fit_transform(X_train)   # learn scaling/encoding from training data only
X_test_processed = preprocessor.transform(X_test)          # apply the same learned transform to test data

print(f"Processed shape  : train={X_train_processed.shape}, test={X_test_processed.shape}")


def evaluate(y_true, y_pred):
    # Small helper so every model in this script is scored the same way.
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

# Untuned SVM with reasonable default-ish settings, used as a reference point
# to measure how much the hyperparameter search actually helps.
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

# Candidate hyperparameter values RandomizedSearchCV will sample from.
param_grid = {
    "C": [0.1, 0.5, 1, 5, 10, 20, 50, 100],
    "kernel": ["rbf", "poly", "sigmoid"],
    "gamma": ["scale", "auto", 0.0001, 0.001, 0.01, 0.1],
    "degree": [2, 3, 4],
    "class_weight": [None, "balanced"]
}

# 3-fold cross-validation split used to score each candidate hyperparameter set.
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Randomly tries 30 hyperparameter combinations and keeps the one with the best CV F1.
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

search.fit(X_train_processed, y_train)   # runs the whole search purely on training data

svm_model = search.best_estimator_                      # the winning SVM, already refit on full training data
svm_pred = svm_model.predict(X_test_processed)           # class predictions on the untouched test set
svm_prob = svm_model.predict_proba(X_test_processed)     # class probabilities on the test set, used later for the hybrid
svm_result = evaluate(y_test, svm_pred)

print(f"Best CV F1      : {search.best_score_:.4f}")
print(f"Best parameters : {search.best_params_}")

print("\nSVM Tuning Results")
print("-" * 55)
print(f"{'Metric':<12}{'Before':>12}{'After':>12}{'Change':>12}")
print("-" * 48)

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

# No hyperparameter search here - Logistic Regression has few knobs and is
# mainly acting as a diverse second opinion for the hybrid, not a competitor.
lr_model = LogisticRegression(
    max_iter=2000,               # extra iterations so it reliably converges
    class_weight="balanced",     # up-weight minority grade classes
    random_state=42,
    n_jobs=-1
)

lr_model.fit(X_train_processed, y_train)

lr_pred = lr_model.predict(X_test_processed)
lr_prob = lr_model.predict_proba(X_test_processed)   # used later for the hybrid blend
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

# Fresh 3-fold split, used only to choose how much to trust SVM vs LR.
weight_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Will hold, for every training row, the probability that row's own held-out fold model produced.
svm_oof_prob = np.zeros((len(X_train_processed), num_classes))
lr_oof_prob = np.zeros((len(X_train_processed), num_classes))

for fold, (train_idx, valid_idx) in enumerate(weight_cv.split(X_train_processed, y_train), start=1):

    # Retrain a fresh copy of the tuned SVM on this fold's training rows only...
    svm_fold = clone(svm_model)
    svm_fold.fit(X_train_processed[train_idx], y_train[train_idx])
    svm_oof_prob[valid_idx] = svm_fold.predict_proba(X_train_processed[valid_idx])   # ...predict on the held-out rows

    # Same idea for Logistic Regression.
    lr_fold = clone(lr_model)
    lr_fold.fit(X_train_processed[train_idx], y_train[train_idx])
    lr_oof_prob[valid_idx] = lr_fold.predict_proba(X_train_processed[valid_idx])

    print(f"Fold {fold}/3 completed")


best_weight = 0.5   # default in case nothing scores higher than -1 below (should never trigger)
best_accuracy = -1

print("\nWeight Evaluation")
print("-" * 45)
print(f"{'SVM':>6}{'LR':>6}{'Accuracy':>12}{'F1':>10}")
print("-" * 45)

# Try several SVM/LR blend ratios and see which one scores best on the out-of-fold predictions.
for svm_weight in [0.5, 0.6, 0.7, 0.8, 0.9]:

    lr_weight = 1 - svm_weight
    hybrid_oof_prob = svm_weight * svm_oof_prob + lr_weight * lr_oof_prob   # weighted average of class probabilities
    hybrid_oof_pred = hybrid_oof_prob.argmax(axis=1)                        # pick the highest-probability class
    hybrid_oof_result = evaluate(y_train, hybrid_oof_pred)

    print(
        f"{svm_weight:>6.1f}"
        f"{lr_weight:>6.1f}"
        f"{hybrid_oof_result['Accuracy']:>12.4f}"
        f"{hybrid_oof_result['F1']:>10.4f}"
    )

    if hybrid_oof_result["Accuracy"] > best_accuracy:   # keep track of the best-performing weight so far
        best_accuracy = hybrid_oof_result["Accuracy"]
        best_weight = svm_weight

lr_weight = 1 - best_weight

print(f"\nSelected weights: SVM={best_weight:.1f}, Logistic Regression={lr_weight:.1f}")
print(f"Best Hybrid CV Accuracy: {best_accuracy:.4f}")


# ============================================================
# 5. Final Hybrid Model
# ============================================================

print("\n" + "-" * 55)
print("5. Final Hybrid Model")
print("-" * 55)

# Apply the chosen weight to the (already computed) test-set probabilities -
# this is the one and only time the hybrid touches the test set.
final_prob = best_weight * svm_prob + lr_weight * lr_prob
final_pred = final_prob.argmax(axis=1)
hybrid_result = evaluate(y_test, final_pred)

print("\nFinal Model Results")
print("-" * 68)
print(f"{'Model':<20}{'Accuracy':>12}{'F1':>12}{'Precision':>12}{'Recall':>12}")
print("-" * 68)

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


# ============================================================
# SAVE
# Kept the same artifact names as your original SVM.py so the
# Streamlit app keeps working without changes if you still want
# to serve the plain SVM. A new svm_lr_hybrid_model.pkl is also
# saved - a single loadable object that reproduces the blended
# (hybrid) predictions on its own.
# ============================================================

# Wrap the two fitted models + chosen weights into one object with the
# same .predict() / .predict_proba() interface as any sklearn model.
hybrid_model = SVMLogisticHybrid(
    svm_model=svm_model,
    lr_model=lr_model,
    svm_weight=best_weight,
    lr_weight=lr_weight
)

# Sanity check: the wrapper's own predictions should exactly match hybrid_result
# above, since it blends the same two probability arrays with the same weights.
hybrid_check_pred = hybrid_model.predict(X_test_processed)
assert (hybrid_check_pred == final_pred).all(), "Hybrid wrapper predictions do not match the manual blend."

joblib.dump(svm_model, os.path.join(ARTIFACTS_DIR, "svm_grade_model.pkl"))                 # the tuned SVM itself
joblib.dump(preprocessor, os.path.join(ARTIFACTS_DIR, "svm_preprocessor.pkl"))             # fitted scaler/encoder
joblib.dump(label_encoder, os.path.join(ARTIFACTS_DIR, "svm_label_encoder.pkl"))           # grade <-> integer mapping
joblib.dump(feature_columns, os.path.join(ARTIFACTS_DIR, "svm_feature_columns.joblib"))    # column order used at inference
joblib.dump(lr_model, os.path.join(ARTIFACTS_DIR, "logistic_regression_grade_model.pkl"))  # the LR model, for reference
joblib.dump(
    {"svm_weight": best_weight, "lr_weight": lr_weight},
    os.path.join(ARTIFACTS_DIR, "svm_lr_hybrid_weights.joblib")   # the chosen blend ratio, for reference
)
# The hybrid itself - reuses the same preprocessor/label_encoder/feature_columns
# saved above, since SVM and LR were both trained on the same processed features.
joblib.dump(hybrid_model, os.path.join(ARTIFACTS_DIR, "svm_lr_hybrid_model.pkl"))

print("\n" + "-" * 55)
print("Artifacts saved successfully")
print("-" * 55)
print("svm_grade_model.pkl")
print("svm_preprocessor.pkl")
print("svm_label_encoder.pkl")
print("svm_feature_columns.joblib")
print("logistic_regression_grade_model.pkl")
print("svm_lr_hybrid_weights.joblib")
print("svm_lr_hybrid_model.pkl")
print("\nTraining complete.")