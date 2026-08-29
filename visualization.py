
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
EDA_FILE = "Students_Performance_Dataset_Clean.csv"

CONFUSION_DIR = "confusion_matrices"
GRAPH_DIR = "Graph"

os.makedirs(CONFUSION_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)


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
    os.path.join(ARTIFACTS_DIR, "X_test_raw.csv")
)

y_test = pd.read_csv(
    os.path.join(ARTIFACTS_DIR, "y_test.csv")
)["Grade"]


# ============================================================
# CREATE CONFUSION MATRICES
# ============================================================

print("\n============================================")
print("CONFUSION MATRICES")
print("============================================")


for model_name, files in models.items():

    print(f"\nCreating {model_name} Confusion Matrix...")

    model = joblib.load(
        os.path.join(ARTIFACTS_DIR, files["model"])
    )

    preprocessor = joblib.load(
        os.path.join(ARTIFACTS_DIR, files["preprocessor"])
    )

    label_encoder = joblib.load(
        os.path.join(ARTIFACTS_DIR, files["encoder"])
    )

    X_test_processed = preprocessor.transform(X_test)

    prediction_encoded = model.predict(X_test_processed)

    predicted_labels = label_encoder.inverse_transform(
        prediction_encoded
    )

    labels = label_encoder.classes_

    cm = confusion_matrix(
        y_test,
        predicted_labels,
        labels=labels
    )

    print("\nTrue Label vs Predicted Label:")
    print(cm)

    plt.figure(figsize=(8, 6))

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

    plt.title(
        f"{model_name} Confusion Matrix",
        fontsize=15,
        fontweight="bold"
    )

    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)

    output_file = os.path.join(
        CONFUSION_DIR,
        f"{model_name.lower()}_confusion_matrix.png"
    )

    plt.tight_layout()
    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print(f"Saved: {output_file}")


# ============================================================
# LOAD CLEAN DATASET FOR EDA
# ============================================================

print("\n============================================")
print("EXPLORATORY DATA ANALYSIS")
print("============================================")

df = pd.read_csv(EDA_FILE)

print("\nDataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# GRADE ORDER
# ============================================================

grade_order = ["A", "B", "C", "D", "F"]

grade_order = [
    grade
    for grade in grade_order
    if grade in df["Grade"].unique()
]


# ============================================================
# GRAPH 1–9 AND GRAPH 12
# COMBINED INTO ONE PNG
# ============================================================

print("\nCreating combined EDA graph...")


fig, axes = plt.subplots(
    5,
    2,
    figsize=(14, 20)
)

axes = axes.flatten()


# ============================================================
# GRAPH 1 - GRADE DISTRIBUTION
# ============================================================

if "Grade" in df.columns:

    sns.countplot(
        data=df,
        x="Grade",
        order=grade_order,
        ax=axes[0]
    )

    axes[0].set_title(
        "Distribution of Student Grade Categories",
        fontsize=13,
        fontweight="bold"
    )

    axes[0].set_xlabel("Grade")
    axes[0].set_ylabel("Number of Students")


# ============================================================
# GRAPH 2 - ATTENDANCE VS GRADE
# ============================================================

if "Attendance (%)" in df.columns:

    sns.boxplot(
        data=df,
        x="Grade",
        y="Attendance (%)",
        order=grade_order,
        ax=axes[1]
    )

    axes[1].set_title(
        "Attendance Distribution Across Grade Categories",
        fontsize=13,
        fontweight="bold"
    )

    axes[1].set_xlabel("Grade")
    axes[1].set_ylabel("Attendance (%)")


# ============================================================
# GRAPH 3 - STUDY HOURS VS GRADE
# ============================================================

if "Study_Hours_per_Week" in df.columns:

    sns.boxplot(
        data=df,
        x="Grade",
        y="Study_Hours_per_Week",
        order=grade_order,
        ax=axes[2]
    )

    axes[2].set_title(
        "Study Hours per Week Across Grade Categories",
        fontsize=13,
        fontweight="bold"
    )

    axes[2].set_xlabel("Grade")
    axes[2].set_ylabel("Study Hours per Week")


# ============================================================
# GRAPH 4 - MIDTERM SCORE VS GRADE
# ============================================================

if "Midterm_Score" in df.columns:

    sns.boxplot(
        data=df,
        x="Grade",
        y="Midterm_Score",
        order=grade_order,
        ax=axes[3]
    )

    axes[3].set_title(
        "Midterm Score Distribution Across Grade Categories",
        fontsize=13,
        fontweight="bold"
    )

    axes[3].set_xlabel("Grade")
    axes[3].set_ylabel("Midterm Score")


# ============================================================
# GRAPH 5 - FINAL SCORE VS GRADE
# ============================================================

if "Final_Score" in df.columns:

    sns.boxplot(
        data=df,
        x="Grade",
        y="Final_Score",
        order=grade_order,
        ax=axes[4]
    )

    axes[4].set_title(
        "Final Score Distribution Across Grade Categories",
        fontsize=13,
        fontweight="bold"
    )

    axes[4].set_xlabel("Grade")
    axes[4].set_ylabel("Final Score")


# ============================================================
# GRAPH 6 - STRESS LEVEL VS GRADE
# ============================================================

if "Stress_Level (1-10)" in df.columns:

    sns.boxplot(
        data=df,
        x="Grade",
        y="Stress_Level (1-10)",
        order=grade_order,
        ax=axes[5]
    )

    axes[5].set_title(
        "Stress Level Across Grade Categories",
        fontsize=13,
        fontweight="bold"
    )

    axes[5].set_xlabel("Grade")
    axes[5].set_ylabel("Stress Level (1-10)")


# ============================================================
# GRAPH 7 - SLEEP HOURS VS GRADE
# ============================================================

if "Sleep_Hours_per_Night" in df.columns:

    sns.boxplot(
        data=df,
        x="Grade",
        y="Sleep_Hours_per_Night",
        order=grade_order,
        ax=axes[6]
    )

    axes[6].set_title(
        "Sleep Hours per Night Across Grade Categories",
        fontsize=13,
        fontweight="bold"
    )

    axes[6].set_xlabel("Grade")
    axes[6].set_ylabel("Sleep Hours per Night")


# ============================================================
# GRAPH 8 - DEPARTMENT VS GRADE
# ============================================================

if "Department" in df.columns:

    department_grade = pd.crosstab(
        df["Department"],
        df["Grade"],
        normalize="index"
    ) * 100

    department_grade = department_grade.reindex(
        columns=grade_order,
        fill_value=0
    )

    department_grade.plot(
        kind="bar",
        stacked=True,
        ax=axes[7]
    )

    axes[7].set_title(
        "Grade Distribution Across Departments",
        fontsize=13,
        fontweight="bold"
    )

    axes[7].set_xlabel("Department")
    axes[7].set_ylabel("Percentage of Students (%)")
    axes[7].legend(
        title="Grade",
        fontsize=8
    )

    axes[7].tick_params(
        axis="x",
        rotation=45
    )


# ============================================================
# GRAPH 9 - FAMILY INCOME VS GRADE
# ============================================================

if "Family_Income_Level" in df.columns:

    income_grade = pd.crosstab(
        df["Family_Income_Level"],
        df["Grade"],
        normalize="index"
    ) * 100

    income_grade = income_grade.reindex(
        columns=grade_order,
        fill_value=0
    )

    income_grade.plot(
        kind="bar",
        stacked=True,
        ax=axes[8]
    )

    axes[8].set_title(
        "Grade Distribution by Family Income Level",
        fontsize=13,
        fontweight="bold"
    )

    axes[8].set_xlabel("Family Income Level")
    axes[8].set_ylabel("Percentage of Students (%)")
    axes[8].legend(
        title="Grade",
        fontsize=8
    )


# ============================================================
# GRAPH 12 - INTERNET ACCESS VS GRADE
# ============================================================

if "Internet_Access_at_Home" in df.columns:

    internet_grade = pd.crosstab(
        df["Internet_Access_at_Home"],
        df["Grade"],
        normalize="index"
    ) * 100

    internet_grade = internet_grade.reindex(
        columns=grade_order,
        fill_value=0
    )

    internet_grade.plot(
        kind="bar",
        stacked=True,
        ax=axes[9]
    )

    axes[9].set_title(
        "Grade Distribution Based on Internet Access at Home",
        fontsize=13,
        fontweight="bold"
    )

    axes[9].set_xlabel("Internet Access at Home")
    axes[9].set_ylabel("Percentage of Students (%)")
    axes[9].legend(
        title="Grade",
        fontsize=8
    )


# ============================================================
# SAVE COMBINED GRAPH
# ============================================================

plt.suptitle(
    "Student Performance Exploratory Data Analysis",
    fontsize=20,
    fontweight="bold",
    y=0.995
)

plt.tight_layout(
    rect=[0, 0, 1, 0.985]
)

combined_output = os.path.join(
    GRAPH_DIR,
    "graph_1_to_9_12_combined.png"
)

plt.savefig(
    combined_output,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print(f"Saved: {combined_output}")


# ============================================================
# GRAPH 10 - CORRELATION HEATMAP
# SEPARATE PNG
# ============================================================

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

correlation_matrix = df[numeric_columns].corr()

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title(
    "Correlation Heatmap of Numerical Features",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "graph_10_correlation_heatmap.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()



# ============================================================
# COMPLETE
# ============================================================

print("\n============================================")
print("ALL GRAPHS CREATED SUCCESSFULLY")
print("============================================")

print("\nConfusion Matrices:")
print(f" - {CONFUSION_DIR}/xgboost_confusion_matrix.png")
print(f" - {CONFUSION_DIR}/ann_confusion_matrix.png")
print(f" - {CONFUSION_DIR}/svm_confusion_matrix.png")

print("\nEDA Graphs:")
print(f" - {GRAPH_DIR}/graph_1_to_9_12_combined.png")
print(f" - {GRAPH_DIR}/graph_10_correlation_heatmap.png")

