# =============================================================================
# Bank Marketing Prediction Web Application
# Machine Learning Assignment 2
# =============================================================================

# Import required libraries
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# =============================================================================
# Streamlit Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)

# =============================================================================
# Application Title
# =============================================================================

st.title("🏦 Bank Marketing Prediction System")

st.markdown("""
This application predicts whether a customer will subscribe to a Term Deposit
using multiple Machine Learning models.

### Features
- 📂 Upload CSV file
- 🤖 Select Machine Learning model
- 📊 View predictions
- 📈 Display evaluation metrics
- 📉 Display confusion matrix
- ⬇ Download prediction results
""")

# =============================================================================
# Define Project Paths
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# =============================================================================
# Load Machine Learning Models
# =============================================================================

@st.cache_resource
def load_models():
    try:
        models = {
            "Logistic Regression": joblib.load(MODEL_DIR / "logistic_regression.pkl"),
            "Decision Tree": joblib.load(MODEL_DIR / "decision_tree.pkl"),
            "K-Nearest Neighbour": joblib.load(MODEL_DIR / "knn.pkl"),
            "Gaussian Naive Bayes": joblib.load(MODEL_DIR / "naive_bayes.pkl"),
            "Random Forest": joblib.load(MODEL_DIR / "random_forest.pkl"),
        }
        return models

    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

# Load all models
models = load_models()

st.success("✅ All Machine Learning models loaded successfully.")

# =============================================================================
# Model Evaluation Metrics
# =============================================================================

model_metrics = {

    "Logistic Regression": {
        "Accuracy": 0.1223,
        "AUC Score": 0.5126,
        "Precision": 0.1208,
        "Recall": 1.0000,
        "F1 Score": 0.2156,
        "MCC Score": 0.0151
    },

    "Decision Tree": {
        "Accuracy": 0.8740,
        "AUC Score": 0.7050,
        "Precision": 0.4782,
        "Recall": 0.4821,
        "F1 Score": 0.4801,
        "MCC Score": 0.4085
    },

    "K-Nearest Neighbour": {
        "Accuracy": 0.8439,
        "AUC Score": 0.5025,
        "Precision": 0.1095,
        "Recall": 0.0412,
        "F1 Score": 0.0599,
        "MCC Score": -0.0075
    },

    "Gaussian Naive Bayes": {
        "Accuracy": 0.8289,
        "AUC Score": 0.8107,
        "Precision": 0.3440,
        "Recall": 0.4610,
        "F1 Score": 0.3940,
        "MCC Score": 0.3012
    },

    "Random Forest": {
        "Accuracy": 0.8983,
        "AUC Score": 0.9064,
        "Precision": 0.6741,
        "Recall": 0.3034,
        "F1 Score": 0.4185,
        "MCC Score": 0.4072
    }
}
# =============================================================================
# Confusion Matrix Data
# =============================================================================

confusion_matrices = {

    "Logistic Regression": [
        [15, 7937],
        [0, 1091]
    ],

    "Decision Tree": [
        [7378, 574],
        [565, 526]
    ],

    "K-Nearest Neighbour": [
        [7586, 366],
        [1046, 45]
    ],

    "Gaussian Naive Bayes": [
        [6993, 959],
        [588, 503]
    ],

    "Random Forest": [
        [7792, 160],
        [760, 331]
    ]
}


# =============================================================================
# Select Machine Learning Model
# =============================================================================

st.header("🤖 Select Machine Learning Model")

selected_model = st.selectbox(
    "Choose a Machine Learning Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbour",
        "Gaussian Naive Bayes",
        "Random Forest"
    )
)

st.info(f"Selected Model: **{selected_model}**")
# =============================================================================
# Upload Test Data
# =============================================================================

st.header("📂 Upload Test Data")

uploaded_file = st.file_uploader(
    "Choose the Test Data CSV file",
    type=["csv"],
    help="Upload the test data for prediction."
)

if uploaded_file is not None:

    test_data = pd.read_csv(uploaded_file)
    # Create a user-friendly copy for display
    display_data = test_data.copy()

    st.success("✅ Test Data uploaded successfully.")

    st.subheader("Test Data Preview")
    # =============================================================================
    # Decode Categorical Columns for User-Friendly Display
    # =============================================================================

    # Job
    display_data["job"] = display_data["job"].replace({
        0: "Admin",
        1: "Blue-collar",
        2: "Entrepreneur",
        3: "Housemaid",
        4: "Management",
        5: "Retired",
        6: "Self-employed",
        7: "Services",
        8: "Student",
        9: "Technician",
        10: "Unemployed",
        11: "Unknown"
    })

    # Marital Status
    display_data["marital"] = display_data["marital"].replace({
        0: "Divorced",
        1: "Married",
        2: "Single"
    })

    # Education
    display_data["education"] = display_data["education"].replace({
        0: "Primary",
        1: "Secondary",
        2: "Tertiary",
        3: "Unknown"
    })

    # Default
    display_data["default"] = display_data["default"].replace({
        0: "No",
        1: "Yes"
    })

    # Housing Loan
    display_data["housing"] = display_data["housing"].replace({
        0: "No",
        1: "Yes"
    })

    # Personal Loan
    display_data["loan"] = display_data["loan"].replace({
        0: "No",
        1: "Yes"
    })

    # Contact Type
    display_data["contact"] = display_data["contact"].replace({
        0: "Cellular",
        1: "Telephone",
        2: "Unknown"
    })

    # Month
    display_data["month"] = display_data["month"].replace({
        0: "January",
        1: "February",
        2: "March",
        3: "April",
        4: "May",
        5: "June",
        6: "July",
        7: "August",
        8: "September",
        9: "October",
        10: "November",
        11: "December"
    })

    # Previous Campaign Outcome
    display_data["poutcome"] = display_data["poutcome"].replace({
        0: "Failure",
        1: "Other",
        2: "Success",
        3: "Unknown"
    })

    

    st.dataframe(display_data)
    # st.subheader("Feature Order in Uploaded Test Data")

    # st.write(test_data.columns.tolist())
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Number of Rows", test_data.shape[0])

    with col2:
        st.metric("Number of Columns", test_data.shape[1])

 # =============================================================================
# Predict using Selected Model
# =============================================================================

if uploaded_file is not None:

    if st.button("🚀 Predict"):

        # Select the chosen model
        model = models[selected_model]

        # Generate predictions
        predictions = model.predict(test_data)

        # Display raw predictions (for debugging)
        # st.subheader("Raw Predictions")
        # st.write(predictions)

        # Create a copy of the uploaded data
        # results = test_data.copy()
        results = display_data.copy()

        # Add prediction column
        results["Prediction"] = predictions

        # Convert numeric predictions into readable labels
        results["Prediction"] = results["Prediction"].replace({
            0: "Not Subscribe",
            1: "Subscribe"
        })

        st.success("✅ Prediction completed successfully.")

        # =============================================================================
        # Prediction Results
        # =============================================================================

        st.subheader("Prediction Results")
        st.dataframe(results)

        # =============================================================================
        # Prediction Summary
        # =============================================================================

        st.subheader("Prediction Summary")

        prediction_counts = results["Prediction"].value_counts()

        st.dataframe(prediction_counts)

        st.bar_chart(prediction_counts)
        
        # =============================================================================
        # Model Used
        # =============================================================================

        st.subheader("Model Used")
        st.write(selected_model)
        # =============================================================================
        # Model Evaluation Metrics
        # =============================================================================

        st.header("📊 Model Evaluation Metrics")

        metrics = model_metrics[selected_model]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Accuracy", f"{metrics['Accuracy']:.2%}")

        with col2:
            st.metric("AUC Score", f"{metrics['AUC Score']:.2%}")

        with col3:
            st.metric("Precision", f"{metrics['Precision']:.2%}")

        col4, col5, col6 = st.columns(3)

        with col4:
            st.metric("Recall", f"{metrics['Recall']:.2%}")

        with col5:
            st.metric("F1 Score", f"{metrics['F1 Score']:.2%}")

        with col6:
            st.metric("MCC Score", f"{metrics['MCC Score']:.4f}")

        # =============================================================================
        # Confusion Matrix
        # =============================================================================

        st.header("📉 Confusion Matrix")

        cm = np.array(confusion_matrices[selected_model])

        cm_df = pd.DataFrame(
        cm,
        index=["Actual: No", "Actual: Yes"],
        columns=["Predicted: No", "Predicted: Yes"]
        )

        st.dataframe(cm_df, use_container_width=True)
        from sklearn.metrics import ConfusionMatrixDisplay

        fig, ax = plt.subplots(figsize=(4, 4))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No", "Yes"]
        )

        disp.plot(
            cmap="Blues",
            ax=ax,
            colorbar=False,
            values_format="d"
        )

        ax.set_title(f"{selected_model}")

        plt.tight_layout()

        st.pyplot(fig, use_container_width=False)

                
        # =============================================================================
        # Download Prediction Results
        # =============================================================================

        csv = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download Prediction Results",
            data=csv,
            file_name="prediction_results.csv",
            mime="text/csv"
        )