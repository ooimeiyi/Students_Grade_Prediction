# STUDENT PERFORMANCE DATASET PREPROCESSING

import json
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split


# Settings
INPUT_FILE = "Students_Performance_Dataset.csv"
OUTPUT_FILE = "Students_Performance_Dataset_Clean.csv"
ARTIFACTS_DIR = "artifacts"
TARGET = "Grade"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Load the dataset into a pandas DataFrame
df = pd.read_csv(INPUT_FILE)

print("STUDENT PERFORMANCE DATASET PREPROCESSING")
print("Original dataset shape:", df.shape)


# Remove duplicates
duplicate_count = df.duplicated().sum()
print("\nDuplicate rows:", duplicate_count)

df = df.drop_duplicates()
print("Shape after removing duplicates:", df.shape)


# Remove personal and unnecessary columns
remove_columns = [
    "Student_ID",
    "StudentID",
    "ID",
    "First_Name",
    "Last_Name",
    "Email",
    "Gender",
    "Age",
    "Parent_Education_Level"
]

remove_columns = [
    column for column in remove_columns
    if column in df.columns
]

df = df.drop(columns=remove_columns)

print("\nRemoved columns:", remove_columns if remove_columns else "None")


# Check missing values before cleaning
print("\nMissing values before cleaning:")
print(df.isnull().sum())


# Remove Total_Score
if "Total_Score" in df.columns:
    df = df.drop(columns=["Total_Score"])
    print("\nTotal_Score: REMOVED")
else:
    print("\nTotal_Score: NOT FOUND")


# Check target
if TARGET not in df.columns:
    raise ValueError("Grade column was not found in dataset.")


# Split data
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# Save train/test data
X_train.to_csv(
    os.path.join(ARTIFACTS_DIR, "X_train_raw.csv"),
    index=False
)

X_test.to_csv(
    os.path.join(ARTIFACTS_DIR, "X_test_raw.csv"),
    index=False
)

y_train.to_frame(name=TARGET).to_csv(
    os.path.join(ARTIFACTS_DIR, "y_train.csv"),
    index=False
)

y_test.to_frame(name=TARGET).to_csv(
    os.path.join(ARTIFACTS_DIR, "y_test.csv"),
    index=False
)


# Save split metadata
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
    os.path.join(ARTIFACTS_DIR, "split_metadata.json"),
    "w",
    encoding="utf-8"
) as file:
    json.dump(split_metadata, file, indent=2)


# Save feature columns
joblib.dump(
    X.columns.tolist(),
    os.path.join(ARTIFACTS_DIR, "feature_columns.joblib")
)


# Save clean dataset
df.to_csv(
    OUTPUT_FILE,
    index=False
)


# Final information
print("\nSaved artifacts:")
print(" - X_train_raw.csv")
print(" - X_test_raw.csv")
print(" - y_train.csv")
print(" - y_test.csv")
print(" - split_metadata.json")
print(" - feature_columns.joblib")

print("\nFinal features:")
for column in X.columns:
    print(" -", column)

print("\nTarget:")
print(" - Grade")


# Check missing values after cleaning
print("\nMissing values after cleaning:")
print(df.isnull().sum())


print("\nGrade distribution:")
print(df[TARGET].value_counts().sort_index())

print("\nClean dataset shape:", df.shape)
print("Saved file:", OUTPUT_FILE)


# Leakage check
print("\nLeakage check:")
print(
    "Total_Score:",
    "REMOVED" if "Total_Score" not in df.columns else "WARNING - PRESENT"
)
print("Grade: TARGET")

print("\nPreprocessing complete.")