import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.svm import SVC  # Ensure SVC is imported if required

# Load the trained model
@st.cache_resource
def load_model():
    model = load('svm_model.joblib')  # Load the trained SVM model
    return model

# Load the saved scaler
@st.cache_resource
def load_scaler():
    scaler = load('scaler.joblib')  # Load the saved scaler
    return scaler

# Streamlit app
def main():
    st.title("SVM Classification App")
    st.write("This app classifies financial data into distress or non-distress categories using an SVM model optimized with PSO.")

    # Sidebar for navigation
    menu = ["Home", "Classify", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Load the model and scaler
    svm_model = load_model()
    scaler = load_scaler()

    if choice == "Home":
        st.subheader("Welcome to the SVM Classification App!")
        st.write("Navigate to the 'Classify' menu to test the model with your own data.")
        
        # Adding instructions on how to use the app
        st.markdown("### How to Use:")
        st.write("""
        1. Navigate to the **Classify** menu from the sidebar.
        2. Input the financial indicators (e.g., NPL, ROA, ROE, etc.) in the provided fields.
        3. Click the **Classify** button to predict whether the data indicates 'Distress' or 'Non-Distress'.
        4. View the prediction results along with the probabilities for each class.
        5. If you encounter any issues, refer to the 'About' section for more information.
        """)
        
    elif choice == "Classify":
        st.subheader("Classify Data")
        st.write("Enter financial indicators below for classification:")

        # Input fields for features
        col1, col2, col3 = st.columns(3)
        with col1:
            npl = st.number_input("NPL", min_value=0.0, format="%.4f")
            roa = st.number_input("ROA", min_value=0.0, format="%.4f")
            roe = st.number_input("ROE", min_value=0.0, format="%.4f")
        with col2:
            nim = st.number_input("NIM", min_value=0.0, format="%.4f")
            bopo = st.number_input("BOPO", min_value=0.0, format="%.4f")
            cir = st.number_input("CIR", min_value=0.0, format="%.4f")
        with col3:
            ldr = st.number_input("LDR", min_value=0.0, format="%.4f")
            car = st.number_input("CAR", min_value=0.0, format="%.4f")
            cr = st.number_input("CR", min_value=0.0, format="%.4f")
            cta = st.number_input("CTA", min_value=0.0, format="%.4f")

        # Collect data in a DataFrame
        input_data = pd.DataFrame(
            [[npl, roa, roe, nim, bopo, cir, ldr, car, cr, cta]],
            columns=["NPL", "ROA", "ROE", "NIM", "BOPO", "CIR", "LDR", "CAR", "CR", "CTA"]
        )

        # Normalize the input data
        try:
            # Ensure the input data is scaled using the loaded scaler
            input_data_scaled = scaler.transform(input_data)

            # Display original and scaled data
            st.write(f"Original Input Data: {input_data}")
            st.write(f"Scaled Input Data: {input_data_scaled}")

            # Set a fixed threshold of 0.3
            threshold = 0.3

            # Predict button
            if st.button("Classify"):
                # Make prediction using the SVM model
                prediction_prob = svm_model.predict_proba(input_data_scaled)
                # Compare the probability of distress (class 1) with the threshold
                if prediction_prob[0][1] >= threshold:
                    prediction = 1  # Distress
                else:
                    prediction = 0  # Non-Distress

                # Display prediction and probability
                st.write(f"Prediction: **{'Distress' if prediction == 1 else 'Non-Distress'}**")

        except Exception as e:
            st.error(f"Error: {e}")

    elif choice == "About":
        st.subheader("About")
        st.write("This app was created to classify financial institutions based on key financial indicators using an SVM model optimized with PSO.")
        st.write("For Further Information Contact maulanaalfin882@gmail.com")
if __name__ == "__main__":
    main()
