# ============================================================
# STUDENT GRADE PREDICTION - STREAMLIT UI
# ============================================================

import os
import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Grade Prediction",
    page_icon="🎓",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("🎓 Student Grade Prediction System")
st.caption("Predict a student's final grade using trained machine learning models.")

# ============================================================
# MODEL FILES
# ============================================================

MODEL_FILES = {
    "XGBoost": "xgboost_grade_model.pkl",
    "ANN": "ann_grade_model.pkl",
    "SVM": "svm_grade_model.pkl"
}

PREPROCESSOR_FILES = {
    "XGBoost": "xgboost_preprocessor.pkl",
    "ANN": "ann_preprocessor.pkl",
    "SVM": "svm_preprocessor.pkl"
}

ENCODER_FILES = {
    "XGBoost": "xgboost_label_encoder.pkl",
    "ANN": "ann_label_encoder.pkl",
    "SVM": "svm_label_encoder.pkl"
}

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_artifacts():
    artifacts = {}
    errors = {}

    for model_name in MODEL_FILES:
        model_path = os.path.join(ARTIFACTS_DIR, MODEL_FILES[model_name])
        preprocessor_path = os.path.join(ARTIFACTS_DIR, PREPROCESSOR_FILES[model_name])
        encoder_path = os.path.join(ARTIFACTS_DIR, ENCODER_FILES[model_name])

        missing = []

        if not os.path.exists(model_path):
            missing.append(MODEL_FILES[model_name])
        if not os.path.exists(preprocessor_path):
            missing.append(PREPROCESSOR_FILES[model_name])
        if not os.path.exists(encoder_path):
            missing.append(ENCODER_FILES[model_name])

        if missing:
            errors[model_name] = missing
            continue

        try:
            artifacts[model_name] = {
                "model": joblib.load(model_path),
                "preprocessor": joblib.load(preprocessor_path),
                "encoder": joblib.load(encoder_path)
            }
        except Exception as e:
            errors[model_name] = [str(e)]

    return artifacts, errors


artifacts, artifact_errors = load_artifacts()

# ============================================================
# ARTIFACT ERROR INFORMATION
# ============================================================

if artifact_errors:
    with st.expander("⚠️ Artifact Problems"):
        for model_name, errors in artifact_errors.items():
            st.write(f"**{model_name}:**")
            for error in errors:
                st.write(f"- {error}")

# ============================================================
# NO MODEL
# ============================================================

if not artifacts:
    st.error("❌ No trained models were found.")
    st.write("Expected artifact folder:")
    st.code(ARTIFACTS_DIR)
    st.write("Run:")
    st.code(
        "python ANN.py\n"
        "python SVM.py\n"
        "python XGBoost.py"
    )
    st.stop()

# ============================================================
# VIEW SELECTION
# ============================================================

current_view = st.sidebar.radio(
    "Choose a view",
    ["Predict Grade", "Compare Models"]
)

# ============================================================
# MODEL COMPARISON
# ============================================================

if current_view == "Compare Models":
    st.header("📊 Model Comparison")

    test_x_path = os.path.join(ARTIFACTS_DIR, "X_test_raw.csv")
    test_y_path = os.path.join(ARTIFACTS_DIR, "y_test.csv")

    if not os.path.exists(test_x_path) or not os.path.exists(test_y_path):
        st.error("X_test_raw.csv or y_test.csv is missing.")
        st.stop()

    X_test = pd.read_csv(test_x_path)
    y_test = pd.read_csv(test_y_path)["Grade"]

    results = []

    for name, artifact in artifacts.items():
        try:
            processed_test = artifact["preprocessor"].transform(X_test)
            encoded_prediction = artifact["model"].predict(processed_test)
            prediction = artifact["encoder"].inverse_transform(encoded_prediction)

            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, prediction),
                "F1 Score": f1_score(y_test, prediction, average="weighted", zero_division=0),
                "Precision": precision_score(y_test, prediction, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, prediction, average="weighted", zero_division=0)
            })
        except Exception as e:
            st.warning(f"{name} could not be evaluated: {e}")

    comparison_df = pd.DataFrame(results)

    if comparison_df.empty:
        st.warning("No models could be evaluated.")
        st.stop()

    # BEST MODEL CARD
    best_model_row = comparison_df.loc[comparison_df["F1 Score"].idxmax()]
    
    st.metric(
        label="🏆 Top Performing Model (Highest F1)", 
        value=best_model_row['Model'], 
        delta=f"F1: {best_model_row['F1 Score']:.4f}"
    )

    # METRICS TABLE
    st.subheader("📋 Evaluation Metrics Summary")
    st.dataframe(
        comparison_df.style.format({
            "Accuracy": "{:.4f}",
            "F1 Score": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}"
        }),
        use_container_width=True
    )

    # NATIVE INTERACTIVE CHARTS
    st.subheader("📈 Interactive Metric Comparison")
    
    metric_tabs = st.tabs(["F1 Score", "Accuracy", "Precision", "Recall"])
    
    chart_data = comparison_df.set_index("Model")
    
    with metric_tabs[0]:
        st.bar_chart(chart_data["F1 Score"])
    with metric_tabs[1]:
        st.bar_chart(chart_data["Accuracy"])
    with metric_tabs[2]:
        st.bar_chart(chart_data["Precision"])
    with metric_tabs[3]:
        st.bar_chart(chart_data["Recall"])

    st.stop()

# ============================================================
# PREDICTION MODEL
# ============================================================

model_name = st.sidebar.selectbox(
    "Choose Prediction Model",
    list(artifacts.keys())
)

selected_model = artifacts[model_name]["model"]
selected_preprocessor = artifacts[model_name]["preprocessor"]
selected_encoder = artifacts[model_name]["encoder"]

st.sidebar.success(f"Active Model: {model_name}")

# ============================================================
# INPUT
# ============================================================

st.header("📝 Student Performance Information")

col1, col2, col3 = st.columns(3)

with col1:
    department = st.selectbox("Department", ["CS", "Engineering", "Business", "Mathematics"])
    attendance = st.number_input("Attendance (%)", 0.0, 100.0, 80.0, 0.1)
    midterm = st.number_input("Midterm Score", 0.0, 100.0, 70.0, 0.1)
    final_score = st.number_input("Final Score", 0.0, 100.0, 70.0, 0.1)
    assignments = st.number_input("Assignments Average", 0.0, 100.0, 70.0, 0.1)

with col2:
    quizzes = st.number_input("Quizzes Average", 0.0, 100.0, 70.0, 0.1)
    participation = st.number_input("Participation Score", 0.0, 100.0, 70.0, 0.1)
    projects = st.number_input("Projects Score", 0.0, 100.0, 70.0, 0.1)
    study_hours = st.number_input("Study Hours per Week", 0.0, 100.0, 10.0, 0.5)
    extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])

with col3:
    internet = st.selectbox("Internet Access at Home", ["Yes", "No"])
    family_income = st.selectbox("Family Income Level", ["Low", "Medium", "High"])
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    sleep = st.number_input("Sleep Hours per Night", 0.0, 24.0, 7.0, 0.1)

# ============================================================
# PREDICT BUTTON
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

    input_data = pd.DataFrame([{
        "Department": department,
        "Attendance (%)": attendance,
        "Midterm_Score": midterm,
        "Final_Score": final_score,
        "Assignments_Avg": assignments,
        "Quizzes_Avg": quizzes,
        "Participation_Score": participation,
        "Projects_Score": projects,
        "Study_Hours_per_Week": study_hours,
        "Extracurricular_Activities": extracurricular,
        "Internet_Access_at_Home": internet,
        "Family_Income_Level": family_income,
        "Stress_Level (1-10)": stress,
        "Sleep_Hours_per_Night": sleep
    }])

    # CHECK INPUT COLUMNS
    try:
        processed_input = selected_preprocessor.transform(input_data)
    except Exception as e:
        st.error("❌ Error preprocessing input.")
        st.exception(e)
        st.write("Input columns:")
        st.write(input_data.columns.tolist())
        st.stop()

    # PREDICT
    try:
        prediction_encoded = selected_model.predict(processed_input)
        predicted_grade = selected_encoder.inverse_transform(prediction_encoded)[0]
    except Exception as e:
        st.error("❌ Prediction failed.")
        st.exception(e)
        st.stop()

    # RESULT CARD
    st.markdown("---")
    st.header("🎯 Prediction Result")

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        with st.container(border=True):
            st.metric(label="Predicted Grade", value=predicted_grade)
            st.caption(f"Engineered by: {model_name}")

    # CONFIDENCE
    if hasattr(selected_model, "predict_proba"):
        try:
            probabilities = selected_model.predict_proba(processed_input)[0]
            class_names = selected_encoder.classes_

            probability_df = pd.DataFrame({
                "Grade": class_names,
                "Probability (%)": probabilities * 100
            }).set_index("Grade")

            with res_col2:
                st.subheader("📊 Class Confidence Breakdown")
                st.bar_chart(probability_df["Probability (%)"])

        except Exception as e:
            st.warning(f"Confidence display unavailable: {e}")

    # STUDENT SUMMARY
    st.subheader("📋 Student Performance Summary")
    
    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.write(f"**Department:** {department}")
        st.write(f"**Attendance:** {attendance:.1f}%")
        st.write(f"**Midterm Score:** {midterm:.1f}")
        st.write(f"**Final Score:** {final_score:.1f}")
        st.write(f"**Assignments:** {assignments:.1f}")
        st.write(f"**Quizzes:** {quizzes:.1f}")
        st.write(f"**Participation:** {participation:.1f}")
        st.write(f"**Projects:** {projects:.1f}")

    with summary_col2:
        st.write(f"**Study Hours:** {study_hours:.1f} hours/week")
        st.write(f"**Extracurricular:** {extracurricular}")
        st.write(f"**Internet Access:** {internet}")
        st.write(f"**Family Income:** {family_income}")
        st.write(f"**Stress Level:** {stress}/10")
        st.write(f"**Sleep:** {sleep:.1f} hours/night")