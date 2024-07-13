import itertools
import time
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Memuat dataset
df = pd.read_csv('Dataset/df_cleaned.csv')

# Memisahkan fitur dan label
X = df.drop(columns=['target'])
y = df['target']

# Melakukan oversampling menggunakan SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# Normalisasi data
scaler = MinMaxScaler()
X_smote = scaler.fit_transform(X_smote)

# Memuat model
model = pickle.load(open('Model/knn_model_normalisasi.pkl', 'rb'))

# Model Evaluation
y_pred = model.predict(X_smote)
accuracy = accuracy_score(y_smote, y_pred)
accuracy = round((accuracy * 100), 2)

# Streamlit
st.set_page_config(
    page_title="Aplikasi Prediksi Penyakit Jantung",
    page_icon=":heart:",
)

st.title("Aplikasi Prediksi Penyakit Jantung")

tab1, tab2 = st.tabs(['Single Prediction', 'Multi Prediction'])

with tab1:
    st.sidebar.header("**Input User** Sidebar")

    # Menambahkan input usia pada sidebar
    age = st.sidebar.number_input(
        label=":white[**Age**]",
        min_value=int(df['age'].min()),
        max_value=int(df['age'].max()),
        step=1
    )

    st.sidebar.write(
        f":orange[Min] value: :orange[**{df['age'].min()}**], :red[Max] value: :red[**{df['age'].max()}**]"
    )
    st.sidebar.write("")

    # Menambahkan input jenis kelamin pada sidebar
    sex_sb = st.sidebar.selectbox(label=":white[**Sex**]", options=["Male", "Female"])
    st.sidebar.write("")
    # Mengubah jenis kelamin menjadi int
    if sex_sb == "Male":
        sex = 1
    elif sex_sb == "Female":
        sex = 0
    
    # Menambahkan input chest pain pada sidebar
    cp_sb = st.sidebar.selectbox(
        label=":white[**Chest Pain**]", 
        options=["Typical Angina",
                 "Atypical Angina", 
                 "Non-anginal Pain", 
                 "Asymptomatic"
                ],
    )
    st.sidebar.write("")
    # Mengubah chest pain menjadi int
    if cp_sb == "Typical Angina":
        cp = 1
    elif cp_sb == "Atypical Angina":
        cp = 2
    elif cp_sb == "Non-anginal Pain":
        cp = 3
    elif cp_sb == "Asymptomatic":
        cp = 4
    
    # Menambahkan input resting blood pressure pada sidebar
    trestbps = st.sidebar.number_input(
        label=":white[**Resting Blood Pressure**]",
        min_value=int(df['trestbps'].min()),
        max_value=int(df['trestbps'].max()),
        step=1
    )
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df['trestbps'].min()}**], :red[Max] value: :red[**{df['trestbps'].max()}**]"
    )
    st.sidebar.write("")

    # Menambahkan input cholesterol pada sidebar
    chol = st.sidebar.number_input(
        label=":white[**Cholesterol**]",
        min_value=int(df['chol'].min()),
        max_value=int(df['chol'].max()),
        step=1
    )
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df['chol'].min()}**], :red[Max] value: :red[**{df['chol'].max()}**]"
    )
    st.sidebar.write("")

    # Menambahkan input fasting blood sugar pada sidebar
    fbs_sb = st.sidebar.selectbox(
        label=":white[**Fasting Blood Sugar > 120 mg/dl**]",
        options=["True", "False"]
    )
    st.sidebar.write("")
    # Mengubah fasting blood sugar menjadi int
    if fbs_sb == "True":
        fbs = 1
    elif fbs_sb == "False":
        fbs = 0

    # Menambahkan input resting electrocardiographic pada sidebar
    restecg_sb = st.sidebar.selectbox(
        label=":white[**Resting Electrocardiographic**]",
        options=["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"
                 ],
    )
    st.sidebar.write("")
    # Mengubah resting electrocardiographic menjadi int
    if restecg_sb == "Normal":
        restecg = 0
    elif restecg_sb == "ST-T wave abnormality":
        restecg = 1
    elif restecg_sb == "Left ventricular hypertrophy":
        restecg = 2

    # Menambahkan input maximum heart rate pada sidebar
    thalach = st.sidebar.number_input(
        label=":white[**Maximum Heart Rate**]",
        min_value=int(df['thalach'].min()),
        max_value=int(df['thalach'].max()),
        step=1
    )
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df['thalach'].min()}**], :red[Max] value: :red[**{df['thalach'].max()}**]"
    )
    st.sidebar.write("")

    # Menambahkan input exercise induced angina pada sidebar
    exang_sb = st.sidebar.selectbox(
        label=":white[**Exercise Induced Angina?**]",
        options=["Yes", "No"]
    )
    st.sidebar.write("")
    # Mengubah exercise induced angina menjadi int
    if exang_sb == "Yes":
        exang = 1
    elif exang_sb == "No":
        exang = 0

    # Menambahkan input ST depression induced by exercise relative to rest pada sidebar
    oldpeak = st.sidebar.number_input(
        label=":white[**ST Depression Induced by Exercise**]",
        min_value=(df['oldpeak'].min()),
        max_value=(df['oldpeak'].max()),   
        step=0.1
    )
    st.sidebar.write(
        f":orange[Min] value: :orange[**{df['oldpeak'].min()}**], :red[Max] value: :red[**{df['oldpeak'].max()}**]"
    )
    st.sidebar.write("")

    # Membuat DataFrame dari input user
    user_input = {
        "Age": [age],
        "Sex": [sex_sb],
        "Chest Pain Type": [cp_sb],
        "Resting Blood Pressure": f"{trestbps} mmHg",
        "Cholesterol": f"{chol} mg/dl",
        "Fasting Blood Sugar > 120 mg/dl?": [fbs_sb],
        "Resting Electrocardiographic": [restecg_sb],
        "Maximum Heart Rate": f"{thalach} bpm",
        "Exercise Induced Angina": [exang_sb],
        "ST Depression Induced by Exercise": [oldpeak]
    }

    previewdf = pd.DataFrame(user_input, index=["User Input"])

    # Menampilkan input user
    st.header("User Input")
    st.write("")
    st.dataframe(previewdf.iloc[:, :6])
    st.dataframe(previewdf.iloc[:, 6:])

    # Menambahkan button untuk prediksi
    st.write("")
    if st.button("Predict"):
        inputs = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]]
        inputs = scaler.transform(inputs)
        prediction = model.predict(inputs)[0]

        # Menampilkan proses
        with st.spinner("Predicting..."):
            time.sleep(1)
            st.success("Prediction Complete")
        
        # Menampilkan hasil prediksi
        if prediction == 0:
            result = ":green[**Healthy**]"
            desc = "Menunjukkan bahwa pengguna tidak memiliki penyakit jantung."
        elif prediction == 1:
            result = ":orange[**Heart Disease level 1**]"
            desc = "Menunjukkan bahwa pengguna memiliki penyakit jantung tingkat ringan. Biasanya diartikan sebagai gejala awal penyakit jantung."
        elif prediction == 2:
            result = ":orange[**Heart Disease level 2**]"
            desc = "Menunjukkan bahwa pengguna memiliki penyakit jantung tingkat sedang. Biasanya diartikan sebagai penyakit jantung yang sudah berkembang. Gejala dan risiko akan bertambah berat jika tidak segera ditangani."
        elif prediction == 3:
            result = ":red[**Heart Disease level 3**]"
            desc = "Menunjukkan bahwa pengguna memiliki penyakit jantung tingkat berat. Biasanya diartikan sebagai penyakit jantung yang sudah parah dan memerlukan penanganan segera. Lebih disarankan segera perawatan medis dan intensif."
        elif prediction == 4:
            result = ":red[**Heart Disease level 4**]"
            desc = "Menunjukkan bahwa pengguna memiliki penyakit jantung tingkat sangat berat. Biasanya diartikan sebagai penyakit jantung yang sudah sangat parah dan mempunyai risiko yang sangat tinggi. Sangat memerlukan perawatan yang sangat intensif."
        
        # Menampilkan hasil prediksi
        st.header("Prediction Result")
        st.write("")
        st.subheader(f"Prediction: {result}")
        st.write(desc)

with tab2:
    st.header("**Multiple Prediction Data**")
    
    # Membuat sample csv data
    sample_data = df.iloc[:5, :-1].to_csv(index=False).encode("utf-8")
    
    st.write("")
    # Menambahkan tombol download
    st.download_button(
        label="Download Sample Data",
        data=sample_data,
        file_name="sample_data.csv",
        mime="text/csv"
    )
    st.write("")

    # Menambahkan button unggah file
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        prediction_arr = model.predict(df)

        # Menampilkan proses
        bar = st.progress(0)
        status_text = st.empty()

        for i in range(1, 70):
            status_text.text(f"Predicting... {i}% Complete")
            bar.progress(i)
            time.sleep(0.01)

        
        # Menyiapkan hasil prediksi
        result_arr = []

        for prediction in prediction_arr:
            if prediction == 0:
                result = "Healthy"
            elif prediction == 1:
                result = "Heart Disease level 1"
            elif prediction == 2:
                result = "Heart Disease level 2"
            elif prediction == 3:
                result = "Heart Disease level 3"
            elif prediction == 4:
                result = "Heart Disease level 4"
            result_arr.append(result)

        uploaded_result = pd.DataFrame({"Prediction": result_arr})

        for i in range(70, 101):
            status_text.text(f"Predicting... {i}% Complete")
            bar.progress(i)
            time.sleep(0.01)
            if i == 100:
                time.sleep(1)
                status_text.empty()
                bar.empty()

        # Menampilkan hasil prediksi
        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(uploaded_result)
        with col2:
            st.dataframe(df)
