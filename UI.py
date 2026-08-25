# ============================================================
# STUDENT GRADE PREDICTION - STREAMLIT UI
# ============================================================

import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from sklearn.model_selection import train_test_split

# ============================================================
# SETTINGS
# ============================================================

ARTIFACTS_DIR = "artifacts"
DATA_FILE = "Students_Performance_Dataset_Clean.csv"

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Grade Prediction",
    page_icon="🎓",
    layout="wide"
)

# ============================================================
# LOAD MODELS AND ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    artifacts = {}

    model_files = {
        "XGBoost": "xgboost_grade_model.pkl",
        "ANN": "ann_grade_model.pkl",
        "SVM": "svm_grade_model.pkl"
    }

    preprocessor_files = {
        "XGBoost": "xgboost_preprocessor.pkl",
        "ANN": "ann_preprocessor.pkl",
        "SVM": "svm_preprocessor.pkl"
    }

    encoder_files = {
        "XGBoost": "xgboost_label_encoder.pkl",
        "ANN": "ann_label_encoder.pkl",
        "SVM": "svm_label_encoder.pkl"
    }

    for model_name in model_files:

        model_path = os.path.join(
            ARTIFACTS_DIR,
            model_files[model_name]
        )

        preprocessor_path = os.path.join(
            ARTIFACTS_DIR,
            preprocessor_files[model_name]
        )

        encoder_path = os.path.join(
            ARTIFACTS_DIR,
            encoder_files[model_name]
        )

        if (
            os.path.exists(model_path)
            and os.path.exists(preprocessor_path)
            and os.path.exists(encoder_path)
        ):

            artifacts[model_name] = {
                "model": joblib.load(model_path),
                "preprocessor": joblib.load(
                    preprocessor_path
                ),
                "encoder": joblib.load(
                    encoder_path
                )
            }

    return artifacts


artifacts = load_artifacts()


@st.cache_data
def evaluate_saved_models():

    """Evaluate the saved artifacts on the same test split used for training."""

    dataset = pd.read_csv(
        DATA_FILE
    )

    features = dataset.drop(
        columns=["Grade"]
    )

    target = dataset["Grade"]

    _, test_features, _, test_target = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target
    )

    results = []

    for name, artifact in artifacts.items():

        encoded_prediction = artifact[
            "model"
        ].predict(
            artifact["preprocessor"].transform(
                test_features
            )
        )

        prediction = artifact[
            "encoder"
        ].inverse_transform(
            encoded_prediction
        )

        results.append({
            "Model": name,

            "Accuracy": accuracy_score(
                test_target,
                prediction
            ),

            "F1 Score": f1_score(
                test_target,
                prediction,
                average="weighted",
                zero_division=0
            ),

            "Precision": precision_score(
                test_target,
                prediction,
                average="weighted",
                zero_division=0
            ),

            "Recall": recall_score(
                test_target,
                prediction,
                average="weighted",
                zero_division=0
            )
        })

    return pd.DataFrame(results)


# ============================================================
# CHECK ARTIFACTS
# ============================================================

if not artifacts:

    st.error(
        "No trained models were found in the artifacts folder."
    )

    st.write(
        "Please run XGBoost.py, ANN.py and SVM.py first."
    )

    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🎓 Model Configuration"
)

current_view = st.sidebar.radio(
    "Choose a view",
    [
        "Predict Grade",
        "Compare Models"
    ],
    key="current_view"
)

show_comparison = (
    current_view == "Compare Models"
)

# ============================================================
# TITLE
# ============================================================

st.title(
    "🎓 Student Grade Prediction System"
)

st.write(
    "Predict a student's final grade using machine learning."
)

# ============================================================
# MODEL COMPARISON
# ============================================================

if show_comparison:

    st.header(
        "📊 Model Comparison"
    )

    st.info(
        "Metrics are calculated from the saved models using the same "
        "20% stratified test split (random_state=42) as the training scripts."
    )

    comparison_df = evaluate_saved_models()

    st.dataframe(
        comparison_df.style.format({
            "Accuracy": "{:.4f}",
            "F1 Score": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}"
        }),
        use_container_width=True
    )

    model_names = [
        "XGBoost",
        "ANN",
        "SVM"
    ]

    short_names = {
        "XGBoost": "XGBoost",
        "ANN": "ANN",
        "SVM": "SVM"
    }

    metric_labels = [
        "Accuracy",
        "F1 Score",
        "Precision",
        "Recall"
    ]

    pastel_colors = [
        "#8ECAE6",
        "#FFB4A2",
        "#BDE0A8"
    ]

    comparison_rows = (
        comparison_df
        .set_index("Model")
        .to_dict("index")
    )

    models_present = [
        model
        for model in model_names
        if model in comparison_rows
    ]

    labels = [
        short_names.get(
            model,
            model
        )
        for model in models_present
    ]

    for row_start in range(
        0,
        len(metric_labels),
        3
    ):

        columns = st.columns(3)

        for column, metric in zip(
            columns,
            metric_labels[
                row_start:row_start + 3
            ]
        ):

            with column:

                st.markdown(
                    f"#### {metric}"
                )

                scores = [
                    comparison_rows[
                        model
                    ][metric]
                    for model in models_present
                ]

                fig, ax = plt.subplots(
                    figsize=(3.2, 3.8)
                )

                bars = ax.bar(
                    labels,
                    scores,
                    color=pastel_colors[
                        :len(models_present)
                    ],
                    width=0.7
                )

                ax.set_ylim(
                    0.0,
                    1.0
                )

                ax.set_yticks([
                    0.0,
                    0.2,
                    0.4,
                    0.6,
                    0.8,
                    1.0
                ])

                ax.set_title(
                    metric,
                    fontsize=11,
                    fontweight="bold",
                    pad=10
                )

                ax.set_ylabel(
                    metric,
                    fontsize=9
                )

                ax.tick_params(
                    axis="x",
                    labelsize=8
                )

                ax.tick_params(
                    axis="y",
                    labelsize=8
                )

                for bar in bars:

                    height = bar.get_height()

                    if not pd.isna(height):

                        ax.annotate(
                            f"{height:.3f}",
                            xy=(
                                bar.get_x()
                                + bar.get_width() / 2,
                                height
                            ),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=8
                        )

                fig.tight_layout()

                st.pyplot(fig)

                plt.close(fig)

    st.markdown("---")

    st.sidebar.info(
        "Choose **Predict Grade** above to return to model selection."
    )

    st.stop()

# ============================================================
# PREDICTION MODEL SELECTION
# ============================================================

model_name = st.sidebar.selectbox(
    "Choose Prediction Model",
    list(artifacts.keys()),
    key="prediction_model"
)

selected_model = artifacts[
    model_name
]["model"]

selected_preprocessor = artifacts[
    model_name
]["preprocessor"]

selected_encoder = artifacts[
    model_name
]["encoder"]

st.sidebar.success(
    f"Active Model: {model_name}"
)

# ============================================================
# STUDENT INPUT
# ============================================================

st.header(
    "📝 Student Information"
)

col1, col2, col3 = st.columns(3)

# ============================================================
# COLUMN 1
# ============================================================

with col1:

    gender = st.selectbox(
        "Gender",
        [
            "Female",
            "Male"
        ]
    )

    age = st.number_input(
        "Age",
        min_value=15,
        max_value=100,
        value=20,
        step=1
    )

    department = st.selectbox(
        "Department",
        [
            "CS",
            "Engineering",
            "Business",
            "Mathematics"
        ]
    )

    attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=0.1
    )

    midterm = st.number_input(
        "Midterm Score",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=0.1
    )

# ============================================================
# COLUMN 2
# ============================================================

with col2:

    final_score = st.number_input(
        "Final Score",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=0.1
    )

    assignments = st.number_input(
        "Assignments Average",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=0.1
    )

    quizzes = st.number_input(
        "Quizzes Average",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=0.1
    )

    participation = st.number_input(
        "Participation Score",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=0.1
    )

    projects = st.number_input(
        "Projects Score",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=0.1
    )

# ============================================================
# COLUMN 3
# ============================================================

with col3:

    study_hours = st.number_input(
        "Study Hours per Week",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=0.5
    )

    extracurricular = st.selectbox(
        "Extracurricular Activities",
        [
            "Yes",
            "No"
        ]
    )

    internet = st.selectbox(
        "Internet Access at Home",
        [
            "Yes",
            "No"
        ]
    )

    parent_education = st.selectbox(
        "Parent Education Level",
        [
            "None",
            "High School",
            "Bachelor's",
            "Master's",
            "PhD"
        ]
    )

    family_income = st.selectbox(
        "Family Income Level",
        [
            "Low",
            "Medium",
            "High"
        ]
    )

    stress = st.slider(
        "Stress Level (1-10)",
        min_value=1,
        max_value=10,
        value=5
    )

    sleep = st.number_input(
        "Sleep Hours per Night",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.1
    )

# ============================================================
# PREDICTION BUTTON
# ============================================================

st.markdown("---")

predict_button = st.button(
    "🔮 Predict Grade",
    type="primary",
    use_container_width=True
)

# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame([{

        "Gender": gender,

        "Age": age,

        "Department": department,

        "Attendance (%)": attendance,

        "Midterm_Score": midterm,

        "Final_Score": final_score,

        "Assignments_Avg": assignments,

        "Quizzes_Avg": quizzes,

        "Participation_Score": participation,

        "Projects_Score": projects,

        "Study_Hours_per_Week": study_hours,

        "Extracurricular_Activities":
            extracurricular,

        "Internet_Access_at_Home":
            internet,

        "Parent_Education_Level":
            parent_education,

        "Family_Income_Level":
            family_income,

        "Stress_Level (1-10)":
            stress,

        "Sleep_Hours_per_Night":
            sleep

    }])

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    try:

        processed_input = (
            selected_preprocessor.transform(
                input_data
            )
        )

    except Exception as e:

        st.error(
            f"Error preprocessing input: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    try:

        prediction_encoded = (
            selected_model.predict(
                processed_input
            )
        )

        predicted_grade = (
            selected_encoder
            .inverse_transform(
                prediction_encoded
            )[0]
        )

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )

        st.stop()

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.markdown("---")

    st.header(
        "🎯 Prediction Result"
    )

    # --------------------------------------------------------
    # GRADE DISPLAY
    # --------------------------------------------------------

    if predicted_grade == "A":

        st.success(
            f"## Predicted Grade: {predicted_grade}"
        )

    elif predicted_grade == "B":

        st.success(
            f"## Predicted Grade: {predicted_grade}"
        )

    elif predicted_grade == "C":

        st.warning(
            f"## Predicted Grade: {predicted_grade}"
        )

    else:

        st.error(
            f"## Predicted Grade: {predicted_grade}"
        )

    st.write(
        f"**Model Used:** {model_name}"
    )

    # ========================================================
    # PREDICTION PROBABILITY
    # ========================================================

    if hasattr(
        selected_model,
        "predict_proba"
    ):

        try:

            probabilities = (
                selected_model
                .predict_proba(
                    processed_input
                )[0]
            )

            class_names = (
                selected_encoder.classes_
            )

            probability_df = pd.DataFrame({
                "Grade": class_names,
                "Probability": probabilities
            })

            probability_df[
                "Probability"
            ] = (
                probability_df[
                    "Probability"
                ] * 100
            )

            st.subheader(
                "Prediction Confidence"
            )

            st.dataframe(
                probability_df.style.format({
                    "Probability": "{:.2f}%"
                }),
                use_container_width=True
            )

            st.bar_chart(
                probability_df.set_index(
                    "Grade"
                )[["Probability"]]
            )

        except Exception:
            pass

    # ========================================================
    # STUDENT SUMMARY
    # ========================================================

    st.subheader(
        "📋 Student Performance Summary"
    )

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:

        st.write(
            f"**Attendance:** {attendance:.1f}%"
        )

        st.write(
            f"**Midterm Score:** {midterm:.1f}"
        )

        st.write(
            f"**Final Score:** {final_score:.1f}"
        )

        st.write(
            f"**Assignments:** {assignments:.1f}"
        )

        st.write(
            f"**Quizzes:** {quizzes:.1f}"
        )

    with summary_col2:

        st.write(
            f"**Projects:** {projects:.1f}"
        )

        st.write(
            f"**Participation:** {participation:.1f}"
        )

        st.write(
            f"**Study Hours:** "
            f"{study_hours:.1f} hours/week"
        )

        st.write(
            f"**Stress Level:** {stress}/10"
        )

        st.write(
            f"**Sleep:** {sleep:.1f} hours/night"
        )