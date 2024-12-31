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
    st.markdown(
    """
    <h1 style="text-align: center;">SVM Classification App</h1>
    """, 
    unsafe_allow_html=True)
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
            npl = st.text_input("NPL", value="")
            roa = st.text_input("ROA", value="")
            roe = st.text_input("ROE", value="")
        with col2:
            nim = st.text_input("NIM", value="")
            bopo = st.text_input("BOPO", value="")
            cir = st.text_input("CIR", value="")
        with col3:
            ldr = st.text_input("LDR", value="")
            car = st.text_input("CAR", value="")
            cr = st.text_input("CR", value="")
            cta = st.text_input("CTA", value="")

        # Validating inputs
        try:
            # Convert inputs to floats if they are not empty
            npl = float(npl) if npl else None
            roa = float(roa) if roa else None
            roe = float(roe) if roe else None
            nim = float(nim) if nim else None
            bopo = float(bopo) if bopo else None
            cir = float(cir) if cir else None
            ldr = float(ldr) if ldr else None
            car = float(car) if car else None
            cr = float(cr) if cr else None
            cta = float(cta) if cta else None
        except ValueError:
            st.error("Please enter valid numbers for all fields.")

        # Collect data in a DataFrame
        input_data = pd.DataFrame(
            [[npl, roa, roe, nim, bopo, cir, ldr, car, cr, cta]],
            columns=["NPL", "ROA", "ROE", "NIM", "BOPO", "CIR", "LDR", "CAR", "CR", "CTA"]
        )

        # Normalize the input data
        try:
            # Ensure the input data is scaled using the loaded scaler
            input_data_scaled = scaler.transform(input_data)

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
        st.markdown("""<h2 style="text-align: center;">Developer Information</h2>""", unsafe_allow_html=True)
        # Layout for photo and biodata
        col1, col2 = st.columns([1, 2])

        with col1:
            # Display the photo
            st.image("profile.jpg", caption="Muhammad Alfin Maulana", width=220)  # Adjust the width as needed

        with col2:
            # Display the biodata
            st.markdown("""
            **Nama:** Muhammad Alfin Maulana  
            **NRP:** 2043211026  
            **Jurusan:** Statistika Bisnis  
            **Institusi:** Institut Teknologi Sepuluh Nopember  
            **No. Telp.:** +62 858 5528 4037  
            **Email:** maulanaalfin882@gmail.com  
            """)

if __name__ == "__main__":
    main()
