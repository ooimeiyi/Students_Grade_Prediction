# ============================================
# STUDENT PERFORMANCE DATASET PREPROCESSING
# FOR GRADE PREDICTION
# ============================================
#
# INPUT:
#   Students_Performance_Dataset.csv
#
# OUTPUT:
#   Students_Performance_Dataset_Clean.csv
#
# TARGET:
#   Grade
#
# IMPORTANT:
#   Final_Score -> KEPT
#   Total_Score -> REMOVED
#
# ============================================

import pandas as pd


# ============================================
# 1. LOAD DATASET
# ============================================

INPUT_FILE = "Students_Performance_Dataset.csv"
OUTPUT_FILE = "Students_Performance_Dataset_Clean.csv"

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
# 3. REMOVE PERSONAL INFORMATION
# ============================================

personal_columns = [
    "Student_ID",
    "StudentID",
    "ID",
    "First_Name",
    "Last_Name",
    "Email"
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

print("\nRemoved personal columns:")

if personal_columns:
    for column in personal_columns:
        print(" -", column)
else:
    print(" - None")


# ============================================
# 4. CHECK MISSING VALUES
# ============================================

print("\n============================================")
print("MISSING VALUES BEFORE CLEANING")
print("============================================")

print(df.isnull().sum())


# ============================================
# 5. HANDLE MISSING VALUES
# ============================================

# Parent Education Level is categorical,
# so fill missing values with its mode.

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
# Final_Score is KEPT.
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

if "Grade" not in df.columns:

    raise ValueError(
        "Grade column was not found in dataset."
    )


# ============================================
# 8. FINAL COLUMN CHECK
# ============================================

print("\n============================================")
print("FINAL FEATURES")
print("============================================")

print("\nFeatures available for Grade prediction:")

for column in df.columns:

    if column != "Grade":

        print(" -", column)


print("\nTarget:")
print(" - Grade")


# ============================================
# 9. SAVE CLEAN DATASET
# ============================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================
# 10. CHECK MISSING VALUES AFTER CLEANING
# ============================================

print("\n============================================")
print("MISSING VALUES AFTER CLEANING")
print("============================================")

print(df.isnull().sum())


# ============================================
# 11. GRADE DISTRIBUTION
# ============================================

print("\n============================================")
print("GRADE DISTRIBUTION")
print("============================================")

print(
    df["Grade"].value_counts()
    .sort_index()
)


# ============================================
# 12. FINAL DATASET INFORMATION
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
    print(" -", column)


# ============================================
# 13. FINAL CHECK
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
        "Total_Score: REMOVED "
    )


if "Final_Score" in df.columns:

    print(
        "Final_Score: INCLUDED "
    )

else:

    print(
        "WARNING: Final_Score is missing!"
    )


print(
    "Grade: TARGET "
)

print("============================================")