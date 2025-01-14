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
    st.title("Prediksi Financial Distress Bank Umum di Indonesia")

    # Sidebar for navigation
    menu = ["Home","Deskripsi Variabel", "Prediksi", "Profil Pembuat"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Load the model and scaler
    svm_model = load_model()
    scaler = load_scaler()

    if choice == "Home":
        st.subheader("Selamat datang di Dashboard Analisis Prediksi Financial Distress!")

    elif choice == "Deskripsi Variabel":
        st.subheader("Penjelasan Variabel")
        st.write("### Non-Performing Loan (NPL)")
        st.write("NPL  merupakan rasio keuangan yang menunjukkan risiko kredit yang dihadapi akibat pemberian kredit dan investasi dana pada portofolio yang berbeda. NPL digunakan untuk menilai potensi kesulitan bank untuk menagih piutangnya yang dapat membahayakan operasional bisnis bank.")

        st.write("### Return on Asset (ROA)")
        st.write("ROA merupakan rasio yang digunakan untuk mengukur seberapa besar jumlah laba bersih yang akan dihasilkan dari setiap rupiah dana yang tertanam dalam total aktiva. ROA merupakan sebuah alat yang memiliki fungsi untuk menilai kemampuan aset bank dalam mendapatkan keuntungan.")

        st.write("### Return on Equity (ROE)")
        st.write("ROE adalah rasio yang menunjukkan daya bank untuk menghasilkan laba atas ekuitas yang diinvestasikan para pemegang saham.")

        st.write("### Net Interest Margin (NIM)")
        st.write("NIM dihitung dengan cara membandingkan antara pendapatan bunga bersih terhadap rata-rata aktiva produktif, kegunaan rasio ini adalah untuk mengetahui kesanggupan manajemen bank dalam mengelola aset produktifnya untuk memperoleh pendapatan bunga bersih. Semakin tinggi NIM sebuah bank maka semakin efektif bank dalam menempatkan aktiva produktif untuk mendorong peningkatan laba.")

        st.write("### Beban Operasi terhadap Pendapatan Operasi (BOPO)")
        st.write("Beban Operasi terhadap Pendapatan Operasi (BOPO adalah rasio efisiensi bank yang mengukur beban operasi terhadap pendapatan operasional, semakin besar nilai BOPO menandakan semakin kurang efisien dalam menjalankan operasional bank.")

        st.write("### Cost to Income Ratio (CIR)")
        st.write("CIR merupakan rasio perbandingan antara biaya dengan pendapatan total. CIR adalah gambaran dari profitabilitas bank, karena melalui CIR dapat diketahui bank tersebut menjalankan usahanya secara efisien atau tidak.")

        st.write("### Loan to Deposit Ratio (LDR)")
        st.write("LDR digunakan untuk mengetahui tingkat likuiditas dari sebuah bank, LDR yang tinggi menandakan semakin besar jumlah dana yang dialirkan kepada debitur daripada ke deposito ataupun tabungan masyarakat.")

        st.write("### Capital Adequacy Ratio (CAR)")
        st.write("CAR atau biasa disebut rasio kecukupan modal menggambarkan rasio kecukupan modal bank yang diperoleh dengan cara membagi total modal dengan aset tertimbang menurut risiko (ATMR).")

        st.write("### Current Ratio (CR)")
        st.write("Current Ratio (CR) digunakan dalam mengukur kemampuan bank untuk memenuhi kewajiban jangka pendeknya yang akan jatuh tempo dengan menggunakan total aset lancar yang ada. CR menggambarkan jumlah ketersediaan aset lancar yang dimiliki dibandingkan dengan total kewajiban lancar.")
    
    elif choice == "Prediksi":
        st.subheader("Data Prediksi")
        st.write("Masukkan Indikator untuk Prediksi:")

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

    elif choice == "Profil Pembuat":
        st.markdown("""<h2 style="text-align: center;">Profil Pembuat</h2>""", unsafe_allow_html=True)
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
