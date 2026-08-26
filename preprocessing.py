# ============================================
# STUDENT PERFORMANCE DATASET PREPROCESSING
# FOR GRADE PREDICTION
# ============================================

# INPUT:
# Students_Performance_Dataset.csv

# OUTPUT:
# Students_Performance_Dataset_Clean.csv

# TARGET:
# Grade

# IMPORTANT:
# Gender -> REMOVED
# Age -> REMOVED
# Final_Score -> KEPT
# Total_Score -> REMOVED

# ============================================

import json
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================
# 1. LOAD DATASET
# ============================================

INPUT_FILE = "Students_Performance_Dataset.csv"
OUTPUT_FILE = "Students_Performance_Dataset_Clean.csv"
ARTIFACTS_DIR = "artifacts"
TARGET = "Grade"

TEST_SIZE = 0.20
RANDOM_STATE = 42

# Load the dataset into a pandas DataFrame
df = pd.read_csv(INPUT_FILE)

print("============================================")
print("STUDENT PERFORMANCE DATASET PREPROCESSING")
print("============================================")

print("\nOriginal dataset shape:")
print(df.shape)

# ============================================
# 2. REMOVE DUPLICATES
# ============================================

duplicate_count = df.duplicated().sum()

print("\nDuplicate rows:", duplicate_count)

df = df.drop_duplicates()

print("Shape after removing duplicates:")
print(df.shape)

# ============================================
# 3. REMOVE PERSONAL / UNNECESSARY INFORMATION
# ============================================

personal_columns = [
    "Student_ID",
    "StudentID",
    "ID",
    "First_Name",
    "Last_Name",
    "Email",
    "Gender",
    "Age"
]

# Only remove columns that exist

personal_columns = [
    column
    for column in personal_columns
    if column in df.columns
]

df = df.drop(
    columns=personal_columns
)

print("\nRemoved personal/unnecessary columns:")

if personal_columns:

    for column in personal_columns:

        print(
            " -",
            column
        )

else:

    print(" - None")

# ============================================
# 4. CHECK MISSING VALUES
# ============================================

print("\n============================================")
print("MISSING VALUES BEFORE CLEANING")
print("============================================")

print(
    df.isnull().sum()
)

# ============================================
# 5. HANDLE MISSING VALUES
# ============================================

# Parent Education Level is categorical,
# so fill missing values with mode.

if "Parent_Education_Level" in df.columns:

    missing_count = (
        df["Parent_Education_Level"]
        .isnull()
        .sum()
    )

    if missing_count > 0:

        mode_value = (
            df["Parent_Education_Level"]
            .mode()[0]
        )

        df["Parent_Education_Level"] = (
            df["Parent_Education_Level"]
            .fillna(mode_value)
        )

        print(
            "\nParent_Education_Level missing values"
        )

        print(
            "Filled with mode:",
            mode_value
        )

# ============================================
# 6. REMOVE TOTAL SCORE
# ============================================

#
# Total_Score is a weighted aggregate of:
#
# Midterm
# Final
# Assignments
# Quizzes
# Participation
# Projects
#
# We DO NOT use Total_Score as an input
# feature when predicting Grade.
#
# Final_Score is KEPT because it is an
# input feature for Grade prediction.
#
# ============================================

if "Total_Score" in df.columns:

    df = df.drop(
        columns=["Total_Score"]
    )

    print(
        "\nRemoved Total_Score."
    )

else:

    print(
        "\nTotal_Score not found."
    )

# ============================================
# 7. CHECK TARGET
# ============================================

if TARGET not in df.columns:

    raise ValueError(
        "Grade column was not found in dataset."
    )

# ============================================
# 8. CREATE AND SAVE REPRODUCIBLE DATA SPLIT
# ============================================

#
# These are raw feature values after dataset
# cleaning, but before model-specific
# preprocessing such as scaling or encoding.
#
# Every training script can use this same split
# for fair comparison.
#
# ============================================

os.makedirs(
    ARTIFACTS_DIR,
    exist_ok=True
)

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# ============================================
# 9. SAVE TRAINING FEATURES
# ============================================

X_train.to_csv(
    os.path.join(
        ARTIFACTS_DIR,
        "X_train_raw.csv"
    ),
    index=False
)

# ============================================
# 10. SAVE TESTING FEATURES
# ============================================

X_test.to_csv(
    os.path.join(
        ARTIFACTS_DIR,
        "X_test_raw.csv"
    ),
    index=False
)

# ============================================
# 11. SAVE TRAINING TARGET
# ============================================

y_train.to_frame(
    name=TARGET
).to_csv(
    os.path.join(
        ARTIFACTS_DIR,
        "y_train.csv"
    ),
    index=False
)

# ============================================
# 12. SAVE TESTING TARGET
# ============================================

y_test.to_frame(
    name=TARGET
).to_csv(
    os.path.join(
        ARTIFACTS_DIR,
        "y_test.csv"
    ),
    index=False
)

# ============================================
# 13. SAVE SPLIT METADATA
# ============================================

split_metadata = {
    "source_file": OUTPUT_FILE,
    "cleaned_from": INPUT_FILE,
    "target": TARGET,
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "stratified": True,
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "feature_count": len(X.columns)
}

with open(
    os.path.join(
        ARTIFACTS_DIR,
        "split_metadata.json"
    ),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        split_metadata,
        file,
        indent=2
    )

# ============================================
# 14. SAVE FEATURE COLUMNS
# ============================================

joblib.dump(
    X.columns.tolist(),
    os.path.join(
        ARTIFACTS_DIR,
        "feature_columns.joblib"
    )
)

print(
    "\nSaved reproducible train/test artifacts:"
)

print(
    " - artifacts/X_train_raw.csv"
)

print(
    " - artifacts/X_test_raw.csv"
)

print(
    " - artifacts/y_train.csv"
)

print(
    " - artifacts/y_test.csv"
)

print(
    " - artifacts/split_metadata.json"
)

print(
    " - artifacts/feature_columns.joblib"
)

# ============================================
# 15. FINAL COLUMN CHECK
# ============================================

print("\n============================================")
print("FINAL FEATURES")
print("============================================")

print(
    "\nFeatures available for Grade prediction:"
)

for column in df.columns:

    if column != TARGET:

        print(
            " -",
            column
        )

print("\nTarget:")

print(
    " - Grade"
)

# ============================================
# 16. SAVE CLEAN DATASET
# ============================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================
# 17. CHECK MISSING VALUES AFTER CLEANING
# ============================================

print("\n============================================")
print("MISSING VALUES AFTER CLEANING")
print("============================================")

print(
    df.isnull().sum()
)

# ============================================
# 18. GRADE DISTRIBUTION
# ============================================

print("\n============================================")
print("GRADE DISTRIBUTION")
print("============================================")

print(
    df[TARGET]
    .value_counts()
    .sort_index()
)

# ============================================
# 19. FINAL DATASET INFORMATION
# ============================================

print("\n============================================")
print("CLEANING COMPLETE")
print("============================================")

print(
    "\nClean dataset shape:",
    df.shape
)

print(
    "\nSaved file:",
    OUTPUT_FILE
)

print("\nFinal columns:")

for column in df.columns:

    print(
        " -",
        column
    )

# ============================================
# 20. FINAL CHECK
# ============================================

print("\n============================================")
print("LEAKAGE CHECK")
print("============================================")

if "Total_Score" in df.columns:

    print(
        "WARNING: Total_Score is still present!"
    )

else:

    print(
        "Total_Score: REMOVED"
    )

if "Final_Score" in df.columns:

    print(
        "Final_Score: INCLUDED"
    )

else:

    print(
        "WARNING: Final_Score is missing!"
    )

if "Gender" in df.columns:

    print(
        "WARNING: Gender is still present!"
    )

else:

    print(
        "Gender: REMOVED"
    )

if "Age" in df.columns:

    print(
        "WARNING: Age is still present!"
    )

else:

    print(
        "Age: REMOVED"
    )

print(
    "Grade: TARGET"
)

print("============================================")