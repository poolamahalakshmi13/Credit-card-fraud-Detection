import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="🔍", layout="wide")
st.title("Credit Card Fraud Detection")


# -------------------------------
# Utility Functions
# -------------------------------
def load_data(file):
    return pd.read_csv(file)


def preprocess_data(data):
    if data.isnull().sum().any():
        data = data.fillna(data.median())
    X = data.drop("Class", axis=1)
    y = data["Class"]
    return X, y


def train_model(X, y, test_size, n_estimators, max_depth):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    return model, scaler, X_test_scaled, y_test


def show_performance(model, X_test, y_test, X):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

    with col2:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance.head(10), x="Importance", y="Feature", ax=ax)
    ax.set_title("Top 10 Most Important Features")
    st.pyplot(fig)


def single_prediction_ui(model, scaler, features):
    st.subheader("Test Single Transaction")
    input_values = []
    cols = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i % 3]:
            val = st.number_input(feature, value=0.0, format="%.6f", key=f"input_{feature}")
            input_values.append(val)

    if st.button("Predict Transaction", type="primary"):
        arr = np.array(input_values).reshape(1, -1)
        arr_scaled = scaler.transform(arr)
        prediction = model.predict(arr_scaled)[0]
        probability = model.predict_proba(arr_scaled)[0]

        if prediction == 1:
            st.error(f"FRAUD DETECTED (Confidence: {probability[1]:.2%})")
        else:
            st.success(f"LEGITIMATE (Confidence: {probability[0]:.2%})")

        c1, c2 = st.columns(2)
        with c1: st.metric("Legitimate Probability", f"{probability[0]:.2%}")
        with c2: st.metric("Fraud Probability", f"{probability[1]:.2%}")


def batch_prediction_ui(model, scaler, features):
    st.subheader("Batch Prediction")
    batch_file = st.file_uploader("Upload batch CSV", type=["csv"], key="batch")
    if batch_file:
        batch = pd.read_csv(batch_file)
        if list(batch.columns) != features:
            st.error("Features in batch file don’t match training data")
        else:
            batch_scaled = scaler.transform(batch)
            preds = model.predict(batch_scaled)
            probs = model.predict_proba(batch_scaled)

            results = batch.copy()
            results["Prediction"] = preds
            results["Fraud_Probability"] = probs[:, 1]
            results["Status"] = results["Prediction"].map({0: "Legitimate", 1: "Fraud"})

            st.write("**Results:**")
            st.dataframe(results, use_container_width=True)

            fraud_count = (preds == 1).sum()
            st.info(f"Found {fraud_count} suspicious transactions out of {len(batch)}")

            csv = results.to_csv(index=False)
            st.download_button("Download Results", csv, "fraud_detection_results.csv", "text/csv")


# -------------------------------
# Main App Flow
# -------------------------------
uploaded_file = st.file_uploader("Upload a CSV file with transactions", type=["csv"])

if uploaded_file:
    try:
        data = load_data(uploaded_file)

        st.subheader(" Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total Records", len(data))
        with col2: st.metric("Features", len(data.columns) - 1 if "Class" in data.columns else len(data.columns))
        with col3:
            if "Class" in data.columns:
                fraud_rate = (data["Class"].sum() / len(data)) * 100
                st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        st.dataframe(data.head(), use_container_width=True)

        if "Class" not in data.columns:
            st.error(" Dataset must have a 'Class' column (0 = Legitimate, 1 = Fraud).")
        else:
            X, y = preprocess_data(data)

            if len(X.select_dtypes(exclude=[np.number]).columns) > 0:
                st.error(" Only numeric columns are allowed.")
            else:
                st.subheader("🔧 Model Training")
                test_size = st.slider("Test Set Size", 0.1, 0.5, 0.3, 0.05)
                n_estimators = st.slider("Number of Trees", 50, 200, 100, 10)
                max_depth = st.slider("Max Depth", 5, 20, 10, 1)

                with st.spinner("Training model..."):
                    model, scaler, X_test_scaled, y_test = train_model(X, y, test_size, n_estimators, max_depth)

                st.subheader("📈 Model Performance")
                show_performance(model, X_test_scaled, y_test, X)

                single_prediction_ui(model, scaler, X.columns.tolist())
                batch_prediction_ui(model, scaler, X.columns.tolist())

    except Exception as e:
        st.error(f" Error: {str(e)}")

else:
    st.info("👆 Please upload a CSV file to get started")
    st.markdown("""
    ### Expected Data Format:
    - Numeric features only  
    - Must include a `Class` column (0 = Legitimate, 1 = Fraud)  
    - Missing values will be filled automatically  
    
    **Example dataset:** [Kaggle Credit Card Fraud Data](https://www.kaggle.com/mlg-ulb/creditcardfraud)
    """)