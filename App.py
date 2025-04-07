# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 16:14:24 2025

@author: Irshad Alam
"""



import pickle
import streamlit as st
from streamlit_option_menu import option_menu

## Load saved models
diabetese_model = pickle.load(open('D:/Machine Learning/Deploying Machine Learning Model/Savemodel/train_model.sav', 'rb'))
heart_disease_model = pickle.load(open('D:/Machine Learning/Deploying Machine Learning Model/Savemodel/heart_disease_model.sav', 'rb'))
parkinson_model = pickle.load(open('D:/Machine Learning/Deploying Machine Learning Model/Savemodel/parkinson_model.sav', 'rb'))

## Apply custom styling
st.markdown("""
    <style>
    body {
        background-color: red
        color: yellow;
        font-family: Arial, sans-serif;
    }
    .stApp {
        background-color: #27548A;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton button {
        background-color: #F6DC43;
        color: black;
        font-size: 18px;
        border-radius: 8px;
        padding: 10px 20px;
        width: 100%;
    }
    .stTextInput>div>div>input {
        background-color: #210F37;
        color: white;
        font-size: 16px;
        border-radius: 5px;
    }
    .diagnosis-success {
        background-color: #d4edda;
        color: green;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-size: 18px;
    }
    .diagnosis-danger {
        background-color: #f8d7da;
        color: red;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

## Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            "Parkinson's Disease Prediction"],
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Diabetes Prediction Page
if selected == "Diabetes Prediction":
    st.title("🔹Diabetes Prediction Using ML")

    # Input data
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input(' Number of Pregnancies')
    with col2:
        Glucose = st.text_input(' Glucose level')
    with col3:
        BloodPressure = st.text_input(' Blood Pressure')

    with col1:
        SkinThickness = st.text_input(' Skin Thickness')
    with col2:
        Insuline = st.text_input(' Insulin Level')
    with col3:
        BMI = st.text_input(' BMI')

    with col1:
        DiabetesPedigreeFunction = st.text_input(' Diabetes Pedigree Function')
    with col2:
        Age = st.text_input(' Age')

    diagnosis = ""

    # Prediction Button
    if st.button('🔍 Check Diabetes'):
        try:
            diab_prediction = diabetese_model.predict([[
                float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness),
                float(Insuline), float(BMI), float(DiabetesPedigreeFunction), float(Age)
            ]])
            diagnosis = "✅ The Person does NOT have Diabetes" if diab_prediction[0] == 0 else "❌ The Person has Diabetes"
            st.markdown(f'<div class="{"diagnosis-success" if diab_prediction[0] == 0 else "diagnosis-danger"}">{diagnosis}</div>', unsafe_allow_html=True)
        except ValueError:
            st.error("❌ Invalid input! Please enter numeric values.")
            
            
            

# Heart Disease Prediction Page
if selected == "Heart Disease Prediction":
    st.title("💖 Heart Disease Prediction Using ML")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input(' Age')
    with col2:
        sex = st.text_input(' Sex (0: Female, 1: Male)')
    with col3:
        cp = st.text_input(' Chest Pain Type')

    with col1:
        trestbps = st.text_input(' Resting Blood Pressure')
    with col2:
        chol = st.text_input(' Serum Cholesterol')
    with col3:
        fbs = st.text_input(' Fasting Blood Sugar')

    with col1:
        restecg = st.text_input(' Resting ECG Results')
    with col2:
        thalach = st.text_input(' Max Heart Rate Achieved')
    with col3:
        exang = st.text_input(' Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input(' ST Depression')
    with col2:
        slope = st.text_input(' Slope of the ST Segment')
    with col3:
        ca = st.text_input(' Major Vessels Colored by Fluoroscopy')

    with col1:
        thal = st.text_input(' Thal (0: Normal, 1: Fixed, 2: Reversible)')

    heart_diagnosis = ""

    if st.button(' Check Heart Disease'):
        try:
            heart_prediction = heart_disease_model.predict([[
                float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs),
                float(restecg), float(thalach), float(exang), float(oldpeak), float(slope),
                float(ca), float(thal)
            ]])
            heart_diagnosis = "✅ The Person does NOT have Heart Disease" if heart_prediction[0] == 0 else "❌ The Person has Heart Disease"
            st.markdown(f'<div class="{"diagnosis-success" if heart_prediction[0] == 0 else "diagnosis-danger"}">{heart_diagnosis}</div>', unsafe_allow_html=True)
        except ValueError:
            st.error("❌ Invalid input! Please enter numeric values.")





# Parkinson's Disease Prediction Page
if selected == "Parkinson's Disease Prediction":
    st.title("🧠 Parkinson's Disease Prediction Using ML")

    col1, col2, col3, col4 = st.columns(4)
    
    ## getting the inpiut from the users
    col1,col2,col3,col4=st.columns(4)
 
    with col1:
        fo=st.text_input('MDVP:Fo(Hz)')
     
    with col2:
        fhi=st.text_input('MDVP:Fhi(Hz)')
    
    with col3:
        flo=st.text_input('MDVP:Flo(Hz)')
    
    with col4:
        jitter_percent=st.text_input('MDVP:Jitter(%)')

    with col1:
        Jitter_Abs=st.text_input('MDVP:Jitter(Abs)')
    
    with col2:
        RAP=st.text_input('MDVP:RAP')
   
    with col3:
        PPQ=st.text_input('MDVP:PPQ')
   
    with col4:
        DDP=st.text_input('Jitter:DDP')
   
    with col1:
        Shimmer=st.text_input('MDVP:Shimmer')
    
    with col2:
        Shimmer_db=st.text_input('MDVP:Shimmer(db)')
   
    with col3:
        APQ3=st.text_input('Shimmer:APQ3')
   
    with col4:
        APQ5=st.text_input('Shimmer:APQ5')
   
    with col1:
        APQ=st.text_input('MDVP:APQ')
    
    with col2:
        DDA=st.text_input('Shimmer:DDA')
   
    with col3:
        NHR=st.text_input('NHR')
   
    with col4:
        HNR=st.text_input('HNR')
   
    with col1:
        RPDE=st.text_input('RPDE')
    
    with col2:
        DFA=st.text_input('DFA')
   
    with col3:
        spread1=st.text_input('spread1')
   
    with col4:
        spread2=st.text_input('spread2')
   
    with col1:
        D2=st.text_input('D2')
     
    with col2:
        PPE=st.text_input('PPE')

    with col3:
    ## code for PPrediction
        parkinsons_diagnosis=''



    if st.button("🧠 Check Parkinson's Disease"):
        try:
            parkinsons_prediction = parkinson_model.predict([[float(fo),float(fhi),float(flo),float(jitter_percent),float(Jitter_Abs),float(RAP),float(PPQ),float(DDP),
                                                              float(Shimmer),float(Shimmer_db),float(APQ3),float(APQ5),float(APQ),float(DDA),float(NHR),float(HNR),float(RPDE),
                                                              float(DFA),float(spread1),float(spread2),float(D2),float(PPE)]])
                                                              
            parkinsons_diagnosis = "✅ The Person does NOT have Parkinson's Disease" if parkinsons_prediction[0] == 0 else "❌ The Person has Parkinson's Disease"
            st.markdown(f'<div class="{"diagnosis-success" if parkinsons_prediction[0] == 0 else "diagnosis-danger"}">{parkinsons_diagnosis}</div>', unsafe_allow_html=True)
        except ValueError:
            st.error("❌ Invalid input! Please enter numeric values.")

