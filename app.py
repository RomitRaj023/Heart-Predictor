'''To run streamilit on localhost
    E:\Anaconda\python.exe -m streamlit run "D:\Heart Predictor\app.py"
type the above in terminal'''

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

#CSS FOR STYLING
st.markdown("""
    <style>
    /* Dark background for the whole page */
    .stApp {
        background-color: #1a1a1a;  /* Dark gray / almost black */
    }

    /* Title styling */
    .stTitle h1 {
        color: #FFD700;  /* Gold color for contrast */
        text-align: center;
    }

    /* Labels of input fields */
    label {
        color: #FFFFFF;  /* White labels for high contrast */
        font-weight: bold;
    }

    /* Input numbers and text font color */
    div.stNumberInput input, div.stTextInput input {
        color: #FFFFFF;  /* White text inside input box */
        background-color: #333333; /* Dark input box */
        border: 1px solid #555555;
    }

    /* Selectbox dropdown text color */
    div.stSelectbox div[role="combobox"] {
        color: #FFFFFF; /* White options text */
        background-color: #333333; /* Dark dropdown background */
        border: 1px solid #555555;
    }

    /* Dropdown hover option */
    div.stSelectbox div[role="option"]:hover {
        background-color: #555555;  /* Lighter gray on hover */
        color: #FFD700;  /* Gold text on hover */
    }

    /* Submit button styling */
    div.stButton > button {
        background-color: #FFD700; /* Gold button */
        color: #1a1a1a;  /* Dark text on button */
        font-weight: bold;
        height: 3em;
        width: 10em;
        border-radius: 10px;
    }

    div.stButton > button:hover {
        background-color: #e6c200;  /* Darker gold on hover */
        color: #1a1a1a;
    }

    /* Subheaders styling */
    h3 {
        color: #FFD700;  /* Gold subheaders */
    }

    /* Tabs styling */
    div[data-baseweb="tab"] {
        color: #FFFFFF;
    }

</style>
""", unsafe_allow_html=True)


st.set_page_config(page_title="Heart Disease Predictor")
st.cache_data.clear()
#st.title("Heart Disease Predictor")
st.markdown('<h1 style="color:#FFD700; text-align:center;">Heart Disease Predictor</h1>', unsafe_allow_html=True)
tab1, tab2 = st.tabs(['Predict', 'Model Information'])

with tab1:
    age = st.number_input("Age (years)", min_value=0, max_value=150)
    
    sex = st.selectbox("Sex", ["Male", "Female"])
    
    chest_pain = st.selectbox("Chest Pain Type", [
        "Typical Angina",
        "Atypical Angina",
        "Non-Anginal Pain",
        "Asymptomatic"
    ])
    
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    
    cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0)
    
    fasting_bs = st.selectbox("Fasting Blood Sugar", [
        "<= 120 mg/dl",
        "> 120 mg/dl"
    ])
    
    resting_ecg = st.selectbox("Resting ECG Results", [
        "Normal",
        "ST-T Wave Abnormality",
        "Left Ventricular Hypertrophy"
    ])
    
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

    # Convert categorical inputs to numeric
    sex = 0 if sex == "Male" else 1
    chest_pain = ["Atypical Angina", "Non-Anginal Pain", "Asymptomatic", "Typical Angina"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope =["Upsloping", "Flat", "Downsloping"].index(st_slope)

    # Create a DataFrame with user inputs
    input_data = pd.DataFrame({
    'Age': [age],
    'Sex': [sex],
    'ChestPainType': [chest_pain],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'FastingBS': [fasting_bs],
    'RestingECG': [resting_ecg],
    'MaxHR': [max_hr],
    'ExerciseAngina': [exercise_angina],
    'Oldpeak': [oldpeak],
    'ST_Slope': [st_slope]
    })
    algonames = ['Decision Trees', 'Logistic Regression', 'Random Forest','Supgort Vector Machine']
    modelnames = ['tree.pkl','LogisticR.pkl', 'RandomForest.pkl','SVM.pkl']
    predictions = []
    def predict_heart_disease(data):
        for modelname in modelnames:
            model = pickle.load(open(modelname,'rb'))
            prediction = model.predict(data)
            predictions.append(prediction)
        return predictions

    # Create a submit button to make predictions
    if st.button("Submit"):
        st.subheader('Results ...')
        st.markdown('-------------------------')
        result = predict_heart_disease(input_data)


        for i in range(len(predictions)):
            st.subheader(algonames[i])
            if result[i][0] == 0:
                st.write("No heart disease detected.")
            else:
                st.write("Heart disease detected.")
            st.markdown('-------------------------')

with tab2:
    import plotly.express as px
    data = {'Decision Trees': 80.97, 'Logistic Regression': 85.86, 'Random Forest': 84.23, 'Support Vector Machine': 84.22, 'GridRF' :89.75}
    Models = list(data.keys())
    Accuracies = list(data.values())
    df = pd.DataFrame(list(zip(Models,Accuracies)),columns=['Models','Accuracies'])
    fig = px.bar(df,y='Accuracies',x='Models')
    st.plotly_chart(fig)