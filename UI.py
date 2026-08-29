# ============================================================
# IMPORT LIBRARIES
# ============================================================

import os
import joblib
import pandas as pd
import altair as alt
import streamlit as st

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# ============================================================
# PROJECT PATHS AND UI SETTINGS
# ============================================================

# Project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Artifacts directory
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# Grade styles
GRADE_STYLE = {
    "A": {"color": "#1DB954", "label": "Excellent"},
    "B": {"color": "#2E9BF0", "label": "Good"},
    "C": {"color": "#F5A623", "label": "Average"},
    "D": {"color": "#E8622C", "label": "At risk"},
    "F": {"color": "#E03C31", "label": "Failing"},
}

# Default grade style
DEFAULT_STYLE = {"color": "#6B7280", "label": ""}

# Model order
MODEL_ORDER = ["ANN", "SVM", "XGBoost"]

# Model colors
MODEL_COLORS = {
    "ANN": "#A8E6B0",
    "SVM": "#FDF1A0",
    "XGBoost": "#A9D6F5",
}

# Default model color
DEFAULT_MODEL_COLOR = "#D1D5DB"


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

# Configure Streamlit page
st.set_page_config(page_title="Student Grade Prediction", page_icon="🎓", layout="wide")

# Display title
st.title("🎓 Student Grade Prediction System")
st.caption("Predict a student's final grade using trained machine learning models.")


# ============================================================
# MODEL AND ARTIFACT FILE NAMES
# ============================================================

# Model files
MODEL_FILES = {
    "XGBoost": "xgboost_grade_model.pkl",
    "ANN": "ann_grade_model.pkl",
    "SVM": "svm_grade_model.pkl"
}

# Preprocessor files
PREPROCESSOR_FILES = {
    "XGBoost": "xgboost_preprocessor.pkl",
    "ANN": "ann_preprocessor.pkl",
    "SVM": "svm_preprocessor.pkl"
}

# Encoder files
ENCODER_FILES = {
    "XGBoost": "xgboost_label_encoder.pkl",
    "ANN": "ann_label_encoder.pkl",
    "SVM": "svm_label_encoder.pkl"
}

# Hybrid model files
HYBRID_RF_FILE = "random_forest_grade_model.pkl"
HYBRID_WEIGHT_FILE = "hybrid_weights.joblib"

# SVM Hybrid model files
SVM_HYBRID_LR_FILE = "logistic_regression_grade_model.pkl"
SVM_HYBRID_WEIGHT_FILE = "svm_lr_hybrid_weights.joblib"

# ============================================================
# LOAD TRAINED MODELS AND ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    # Store artifacts and errors
    artifacts = {}
    errors = {}

    # XGBoost artifact paths
    xgb_model_path = os.path.join(ARTIFACTS_DIR, MODEL_FILES["XGBoost"])
    xgb_preprocessor_path = os.path.join(ARTIFACTS_DIR, PREPROCESSOR_FILES["XGBoost"])
    xgb_encoder_path = os.path.join(ARTIFACTS_DIR, ENCODER_FILES["XGBoost"])
    rf_model_path = os.path.join(ARTIFACTS_DIR, HYBRID_RF_FILE)
    hybrid_weight_path = os.path.join(ARTIFACTS_DIR, HYBRID_WEIGHT_FILE)

    # Check missing XGBoost files
    xgb_missing = []

    if not os.path.exists(xgb_model_path):
        xgb_missing.append(MODEL_FILES["XGBoost"])

    if not os.path.exists(xgb_preprocessor_path):
        xgb_missing.append(PREPROCESSOR_FILES["XGBoost"])

    if not os.path.exists(xgb_encoder_path):
        xgb_missing.append(ENCODER_FILES["XGBoost"])

    if not os.path.exists(rf_model_path):
        xgb_missing.append(HYBRID_RF_FILE)

    if not os.path.exists(hybrid_weight_path):
        xgb_missing.append(HYBRID_WEIGHT_FILE)

    # Record missing XGBoost files
    if xgb_missing:
        errors["XGBoost"] = xgb_missing

    else:

        try:

            # Load XGBoost hybrid artifacts
            artifacts["XGBoost"] = {
                "type": "hybrid",
                "xgb_model": joblib.load(xgb_model_path),
                "rf_model": joblib.load(rf_model_path),
                "preprocessor": joblib.load(xgb_preprocessor_path),
                "encoder": joblib.load(xgb_encoder_path),
                "weights": joblib.load(hybrid_weight_path)
            }

        except Exception as e:
            errors["XGBoost"] = [str(e)]

    # ------------------------------------------------------
    # SVM + Logistic Regression hybrid
    # ------------------------------------------------------
 
    svm_model_path = os.path.join(ARTIFACTS_DIR, MODEL_FILES["SVM"])
    svm_preprocessor_path = os.path.join(ARTIFACTS_DIR, PREPROCESSOR_FILES["SVM"])
    svm_encoder_path = os.path.join(ARTIFACTS_DIR, ENCODER_FILES["SVM"])
    lr_model_path = os.path.join(ARTIFACTS_DIR, SVM_HYBRID_LR_FILE)
    svm_hybrid_weight_path = os.path.join(ARTIFACTS_DIR, SVM_HYBRID_WEIGHT_FILE)
 
    # Check missing SVM files
    svm_missing = []
 
    if not os.path.exists(svm_model_path):
        svm_missing.append(MODEL_FILES["SVM"])
 
    if not os.path.exists(svm_preprocessor_path):
        svm_missing.append(PREPROCESSOR_FILES["SVM"])
 
    if not os.path.exists(svm_encoder_path):
        svm_missing.append(ENCODER_FILES["SVM"])
 
    if not os.path.exists(lr_model_path):
        svm_missing.append(SVM_HYBRID_LR_FILE)
 
    if not os.path.exists(svm_hybrid_weight_path):
        svm_missing.append(SVM_HYBRID_WEIGHT_FILE)
 
    # Record missing SVM files
    if svm_missing:
        errors["SVM"] = svm_missing
 
    else:
 
        try:
 
            # Load SVM hybrid artifacts
            artifacts["SVM"] = {
                "type": "hybrid",
                "svm_model": joblib.load(svm_model_path),
                "lr_model": joblib.load(lr_model_path),
                "preprocessor": joblib.load(svm_preprocessor_path),
                "encoder": joblib.load(svm_encoder_path),
                "weights": joblib.load(svm_hybrid_weight_path)
            }
 
        except Exception as e:
            errors["SVM"] = [str(e)]
 
    # ------------------------------------------------------
    # ANN (single model, unchanged)
    # ------------------------------------------------------
 
    for model_name in ["ANN"]:
 
        # Build artifact paths
        model_path = os.path.join(ARTIFACTS_DIR, MODEL_FILES[model_name])
        preprocessor_path = os.path.join(ARTIFACTS_DIR, PREPROCESSOR_FILES[model_name])
        encoder_path = os.path.join(ARTIFACTS_DIR, ENCODER_FILES[model_name])
 
        # Check missing files
        missing = []
 
        if not os.path.exists(model_path):
            missing.append(MODEL_FILES[model_name])
 
        if not os.path.exists(preprocessor_path):
            missing.append(PREPROCESSOR_FILES[model_name])
 
        if not os.path.exists(encoder_path):
            missing.append(ENCODER_FILES[model_name])
 
        # Skip model if files are missing
        if missing:
            errors[model_name] = missing
            continue
 
        try:
 
            # Load model artifacts
            artifacts[model_name] = {
                "type": "single",
                "model": joblib.load(model_path),
                "preprocessor": joblib.load(preprocessor_path),
                "encoder": joblib.load(encoder_path)
            }
 
        except Exception as e:
            errors[model_name] = [str(e)]
 
    # Return loaded artifacts and errors
    return artifacts, errors

# Load all artifacts
artifacts, artifact_errors = load_artifacts()


# ============================================================
# DISPLAY ARTIFACT ERRORS
# ============================================================

# Display artifact errors
if artifact_errors:

    with st.expander("⚠️ Artifact Problems"):

        for model_name, errors in artifact_errors.items():

            st.write(f"**{model_name}:**")

            for error in errors:
                st.write(f"- {error}")


# Stop if no models are available
if not artifacts:

    st.error("❌ No trained models were found.")

    st.write("Expected artifact folder:")
    st.code(ARTIFACTS_DIR)

    st.write("Run:")
    st.code("python ANN.py\npython SVM.py\npython XGBoost.py")

    st.stop()


# ============================================================
# SIDEBAR VIEW SELECTION
# ============================================================

# Select application view
current_view = st.sidebar.radio("Choose a view", ["Predict Grade", "Compare Models"])


# ============================================================
# MODEL COMPARISON PAGE
# ============================================================

if current_view == "Compare Models":

    st.header("📊 Model Comparison")

    # Test dataset paths
    test_x_path = os.path.join(ARTIFACTS_DIR, "X_test_raw.csv")
    test_y_path = os.path.join(ARTIFACTS_DIR, "y_test.csv")

    # Check test files
    if not os.path.exists(test_x_path) or not os.path.exists(test_y_path):
        st.error("X_test_raw.csv or y_test.csv is missing.")
        st.stop()

    # Load test data
    X_test = pd.read_csv(test_x_path)
    y_test = pd.read_csv(test_y_path)["Grade"]

    # Store results
    results = []

    # Evaluate models
    for name, artifact in artifacts.items():

        try:

            # Preprocess test data
            processed_test = artifact["preprocessor"].transform(X_test)

            # Single model prediction
            if artifact["type"] == "single":

                encoded_prediction = artifact["model"].predict(processed_test)
                prediction = artifact["encoder"].inverse_transform(encoded_prediction)

            # Hybrid prediction (works for XGB+RF, SVM+LR, or any future
            # two-model hybrid, without hardcoding component names).
            #
            # Each hybrid artifact stores its component models under keys
            # ending in "_model" (e.g. "xgb_model"/"rf_model" or
            # "svm_model"/"lr_model"), and matching weights under
            # "<prefix>_weight" inside artifact["weights"]. This block
            # discovers those keys instead of assuming XGB/RF specifically.
            else:

                model_keys = sorted(
                    key for key in artifact
                    if key.endswith("_model")
                )

                if len(model_keys) != 2:
                    raise ValueError(
                        f"Expected 2 component models for a hybrid, "
                        f"found {len(model_keys)}: {model_keys}"
                    )

                hybrid_prob = None

                for model_key in model_keys:

                    prefix = model_key[: -len("_model")]
                    weight_key = f"{prefix}_weight"

                    if weight_key not in artifact["weights"]:
                        raise KeyError(
                            f"Missing '{weight_key}' in weights for "
                            f"'{model_key}'."
                        )

                    component_prob = artifact[model_key].predict_proba(processed_test)
                    component_weight = artifact["weights"][weight_key]

                    weighted_prob = component_weight * component_prob

                    hybrid_prob = (
                        weighted_prob
                        if hybrid_prob is None
                        else hybrid_prob + weighted_prob
                    )

                # Select highest probability class
                hybrid_prediction_encoded = hybrid_prob.argmax(axis=1)
                prediction = artifact["encoder"].inverse_transform(hybrid_prediction_encoded)

            # Calculate metrics
            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, prediction),
                "F1 Score": f1_score(y_test, prediction, average="weighted", zero_division=0),
                "Precision": precision_score(y_test, prediction, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, prediction, average="weighted", zero_division=0)
            })

        except Exception as e:
            st.warning(f"{name} could not be evaluated: {e}")


    # Convert results to DataFrame
    comparison_df = pd.DataFrame(results)

    # Stop if no models can be evaluated
    if comparison_df.empty:
        st.warning("No models could be evaluated.")
        st.stop()


    # Find best model
    best_model_row = comparison_df.loc[comparison_df["Accuracy"].idxmax()]

    # Display best model
    st.metric(
        label="🏆 Top Performing Model (Highest Accuracy)",
        value=best_model_row["Model"],
        delta=f"Accuracy: {best_model_row['Accuracy']:.4f}"
    )
    
    # Arrange models in fixed order
    model_order = ["ANN", "SVM", "XGBoost"]

    comparison_df["Model"] = pd.Categorical(
        comparison_df["Model"],
        categories=model_order,
        ordered=True
    )

    comparison_df = comparison_df.sort_values("Model").reset_index(drop=True)
    
    comparison_df.index = range(1, len(comparison_df) + 1)

    # Display metrics
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

    # ========================================================
    # METRIC COMPARISON CHARTS
    # ========================================================

    st.subheader("📈 Metric Comparison")

    # Create metric tabs
    metric_tabs = st.tabs(["Accuracy", "F1 Score", "Precision", "Recall"])

    # Get available models
    models_present = comparison_df["Model"].tolist()

    # Arrange models in predefined order
    ordered_models = (
        [m for m in MODEL_ORDER if m in models_present] +
        [m for m in models_present if m not in MODEL_ORDER]
    )

    # Set chart colors
    color_domain = ordered_models

    color_range = [MODEL_COLORS.get(m, DEFAULT_MODEL_COLOR) for m in ordered_models]

    # Create metric chart
    def render_metric_chart(metric_column):

        chart = (
            alt.Chart(comparison_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Model:N", title=None, sort=ordered_models),
                y=alt.Y(f"{metric_column}:Q", title=metric_column, scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "Model:N",
                    scale=alt.Scale(domain=color_domain, range=color_range),
                    legend=None
                ),
                tooltip=["Model", alt.Tooltip(f"{metric_column}:Q", format=".4f")]
            )
            .properties(height=320)
        )

        # Display chart
        st.altair_chart(chart, use_container_width=True)


    # Display metric charts
    with metric_tabs[0]:
        render_metric_chart("Accuracy")

    with metric_tabs[1]:
        render_metric_chart("F1 Score")

    with metric_tabs[2]:
        render_metric_chart("Precision")

    with metric_tabs[3]:
        render_metric_chart("Recall")

    # Stop after comparison page
    st.stop()


# ============================================================
# PREDICTION MODEL SELECTION
# ============================================================

# Select prediction model
model_name = st.sidebar.selectbox("Choose Prediction Model", list(artifacts.keys()))

# Get selected model artifacts
selected_artifact = artifacts[model_name]

selected_preprocessor = selected_artifact["preprocessor"]
selected_encoder = selected_artifact["encoder"]

# Display active model
st.sidebar.success(f"Active Model: {model_name}")


# ============================================================
# STUDENT INPUT SECTION
# ============================================================

st.header("📝 Student Performance Information")

# Create input columns
col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# Column 1
# ------------------------------------------------------------

with col1:

    department = st.selectbox("Department", ["CS", "Engineering", "Business", "Mathematics"])
    attendance = st.number_input("Attendance (%)", 0.0, 100.0, 80.0, 0.1)
    midterm = st.number_input("Midterm Score", 0.0, 100.0, 70.0, 0.1)
    final_score = st.number_input("Final Score", 0.0, 100.0, 70.0, 0.1)
    assignments = st.number_input("Assignments Average", 0.0, 100.0, 70.0, 0.1)


# ------------------------------------------------------------
# Column 2
# ------------------------------------------------------------

with col2:

    quizzes = st.number_input("Quizzes Average", 0.0, 100.0, 70.0, 0.1)
    participation = st.number_input("Participation Score", 0.0, 100.0, 70.0, 0.1)
    projects = st.number_input("Projects Score", 0.0, 100.0, 70.0, 0.1)
    study_hours = st.number_input("Study Hours per Week", 0.0, 100.0, 10.0, 0.5)
    extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])


# ------------------------------------------------------------
# Column 3
# ------------------------------------------------------------

with col3:

    internet = st.selectbox("Internet Access at Home", ["Yes", "No"])
    family_income = st.selectbox("Family Income Level", ["Low", "Medium", "High"])
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    sleep = st.number_input("Sleep Hours per Night", 0.0, 24.0, 7.0, 0.1)


# Add separator
st.markdown("---")

# Prediction button
predict_button = st.button("🔮 Predict Grade", type="primary", use_container_width=True)


# ============================================================
# PREDICTION PROCESS
# ============================================================

# Run prediction when button is clicked
if predict_button:

    # Create input DataFrame
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


    # ========================================================
    # PREPROCESS USER INPUT
    # ========================================================

    try:

        # Apply preprocessing pipeline
        processed_input = selected_preprocessor.transform(input_data)

    except Exception as e:

        st.error("❌ Error preprocessing input.")
        st.exception(e)

        st.write("Input columns:")
        st.write(input_data.columns.tolist())

        st.stop()


    # ========================================================
    # GENERATE GRADE PREDICTION
    # ========================================================

    try:

        # Single model prediction
        if selected_artifact["type"] == "single":

            prediction_encoded = selected_artifact["model"].predict(processed_input)
            predicted_grade = selected_artifact["encoder"].inverse_transform(prediction_encoded)[0]

        else:

            model_keys = sorted(
                key for key in selected_artifact
                if key.endswith("_model")
            )

            if len(model_keys) != 2:
                raise ValueError(
                    f"Expected 2 component models for a hybrid, "
                    f"found {len(model_keys)}: {model_keys}"
                )

            hybrid_probabilities = None

            for model_key in model_keys:

                prefix = model_key[: -len("_model")]
                weight_key = f"{prefix}_weight"

                if weight_key not in selected_artifact["weights"]:
                    raise KeyError(
                        f"Missing '{weight_key}' in weights for "
                        f"'{model_key}'."
                    )

                component_prob = selected_artifact[model_key].predict_proba(processed_input)[0]
                component_weight = selected_artifact["weights"][weight_key]

                weighted_prob = component_weight * component_prob

                hybrid_probabilities = (
                    weighted_prob
                    if hybrid_probabilities is None
                    else hybrid_probabilities + weighted_prob
                )

            # Select highest probability class
            prediction_encoded = [hybrid_probabilities.argmax()]

            predicted_grade = selected_artifact["encoder"].inverse_transform(prediction_encoded)[0]

    except Exception as e:

        st.error("❌ Prediction failed.")
        st.exception(e)
        st.stop()

    # ========================================================
    # CALCULATE CLASS PROBABILITIES / CONFIDENCE
    # ========================================================

    probability_df = None
    top_confidence = None

    try:

        # Get single model probabilities
        if selected_artifact["type"] == "single":

            if hasattr(selected_artifact["model"], "predict_proba"):
                probabilities = selected_artifact["model"].predict_proba(processed_input)[0]

        # Use hybrid probabilities
        else:
            probabilities = hybrid_probabilities


        # Get grade classes
        class_names = selected_encoder.classes_

        # Create probability DataFrame
        probability_df = pd.DataFrame({
            "Grade": class_names,
            "Probability": probabilities
        }).sort_values("Probability", ascending=False).reset_index(drop=True)


        # Get predicted grade confidence
        top_confidence = probability_df.loc[
            probability_df["Grade"] == predicted_grade,
            "Probability"
        ].iloc[0]

    except Exception as e:

        st.warning(f"Confidence display unavailable: {e}")


    # ========================================================
    # DISPLAY PREDICTION RESULT
    # ========================================================

    st.markdown("---")
    st.header("🎯 Prediction Result")

    # Get predicted grade style
    style = GRADE_STYLE.get(str(predicted_grade).upper(), DEFAULT_STYLE)


    # Create result columns
    res_col1, res_col2 = st.columns([1, 2], gap="large")


    # --------------------------------------------------------
    # PREDICTED GRADE DISPLAY
    # --------------------------------------------------------

    with res_col1:

        st.subheader("Predicted Grade")

        # Display grade based on category
        if predicted_grade == "A":
            st.success(f"## {predicted_grade}")

        elif predicted_grade == "B":
            st.info(f"## {predicted_grade}")

        elif predicted_grade == "C":
            st.warning(f"## {predicted_grade}")

        elif predicted_grade == "D":
            st.warning(f"## {predicted_grade}")

        elif predicted_grade == "F":
            st.error(f"## {predicted_grade}")

        else:
            st.write(f"## {predicted_grade}")

        # Display grade description
        if style["label"]:
            st.write(f"**{style['label']}**")

        # Display confidence
        if top_confidence is not None:
            st.metric("Confidence", f"{top_confidence * 100:.1f}%")

        # Display selected model
        st.caption(f"{model_name}")


    # --------------------------------------------------------
    # CONFIDENCE BREAKDOWN CHART
    # --------------------------------------------------------

    with res_col2:

        st.subheader("📊 Class Confidence Breakdown")

        # Display chart if probability data exists
        if probability_df is not None:

            chart_df = probability_df.copy()

            # Convert probability to percentage
            chart_df["Probability (%)"] = chart_df["Probability"] * 100

            # Mark predicted grade
            chart_df["Predicted"] = chart_df["Grade"] == predicted_grade

            # Define grade order
            grade_order = [g for g in GRADE_STYLE if g in chart_df["Grade"].values] or None


            # Create confidence chart
            confidence_chart = (
                alt.Chart(chart_df)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X(
                        "Grade:N",
                        sort=grade_order,
                        title=None,
                        axis=alt.Axis(labelAngle=0)
                    ),
                    y=alt.Y(
                        "Probability (%):Q",
                        title="Probability (%)",
                        scale=alt.Scale(domain=[0, 100])
                    ),
                    color=alt.condition(
                        alt.datum.Predicted,
                        alt.value(style["color"]),
                        alt.value("#D1D5DB")
                    ),
                    tooltip=[
                        "Grade",
                        alt.Tooltip("Probability (%):Q", format=".1f")
                    ]
                )
            )


            # Create probability labels
            labels = (
                alt.Chart(chart_df)
                .mark_text(dy=-8, fontWeight="bold", color="#FFFFFF")
                .encode(
                    x=alt.X("Grade:N", sort=grade_order, axis=alt.Axis(labelAngle=90)),
                    y="Probability (%):Q",
                    text=alt.Text("Probability (%):Q", format=".1f")
                )
            )


            # Display chart and labels
            st.altair_chart(
                (confidence_chart + labels).properties(height=320),
                use_container_width=True
            )

        else:

            st.info("This model does not expose class probabilities.")


    # ========================================================
    # STUDENT PERFORMANCE SUMMARY
    # ========================================================

    # Display student summary
    with st.expander("📋 Student Performance Summary", expanded=False):

        # Create summary columns
        summary_col1, summary_col2 = st.columns(2)


        # ----------------------------------------------------
        # SUMMARY COLUMN 1
        # ----------------------------------------------------

        with summary_col1:

            st.write(f"**Department:** {department}")
            st.write(f"**Attendance:** {attendance:.1f}%")
            st.write(f"**Midterm Score:** {midterm:.1f}")
            st.write(f"**Final Score:** {final_score:.1f}")
            st.write(f"**Assignments:** {assignments:.1f}")
            st.write(f"**Quizzes:** {quizzes:.1f}")
            st.write(f"**Participation:** {participation:.1f}")


        # ----------------------------------------------------
        # SUMMARY COLUMN 2
        # ----------------------------------------------------

        with summary_col2:

            st.write(f"**Projects:** {projects:.1f}")
            st.write(f"**Study Hours:** {study_hours:.1f} hours/week")
            st.write(f"**Extracurricular:** {extracurricular}")
            st.write(f"**Internet Access:** {internet}")
            st.write(f"**Family Income:** {family_income}")
            st.write(f"**Stress Level:** {stress}/10")
            st.write(f"**Sleep:** {sleep:.1f} hours/night")