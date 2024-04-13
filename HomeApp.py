import streamlit as st
import pandas as pd
import json
from streamlit_lottie import st_lottie
from st_pages import show_pages, Page
import time
from Models import ClassifierModelsHeart
import pickle
from PIL import Image
from streamlit_toggle import st_toggle_switch
import numpy as np

#page configuration
st.set_page_config(
    page_title="Heart Disease Track",
    page_icon=":anatomical_heart:",
    layout="wide"
)
#pages
show_pages([
            Page("HomeApp.py", "Home"),
            Page("pages/ModelsHeart.py", "Models")
        ])
#CSS streamlit style
with open ("app_style/streamlit.css") as f:
    st.markdown(f'<style> {f.read()} </style>', unsafe_allow_html=True)

#lottiefile loader
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
#data loader
def read_data():
    delimiter = ';'
    df = pd.read_csv("dataset/HeartDataset.csv", delimiter=delimiter)
    df['Oldpeak'] = df['Oldpeak'].str.replace(',', '.').astype(float)
    return df

#files
image = Image.open('png/anatomical-heart_1fac0.png')

#sidebar
with st.sidebar:
    st.image(image)
    st.title("Heart Disease Track")
    st.header('Please, fill your informations to predict your heart condition')
    Age = st.slider('Age', 28, 77)
    Sex = st.selectbox('Sex', ('M', 'F'))
    ChestPainType = st.selectbox('Chest Pain Type', ('ASY', 'ATA', 'NAP', 'TA'), help="ATA - atypical angina, NAP - non-anginal pain, ASY - asymptomatic")
    RestingBP = st.sidebar.slider('Resting Blood Pressure', 0, 200, help="Blood pressure at rest.")
    Cholesterol = st.slider('Cholesterol', 0, 600,help="Cholesterol level [mg/dl]")
    FastingBS = st.slider('Fasting Blood Sugar', 0, 1, help="0 - if fasting blood sugar is below 120 mg/dl, 1 - if fasting blood sugar is 120 mg/dl or higher")
    RestingECG = st.selectbox('Resting Electrocardiogram', ('Normal', 'ST', 'LVH'), help="Normal - normal result, ST - abnormal result, LVH -Left Ventricular Hypertrophy")
    MaxHR = st.slider('Maximum Heart Rate', 60, 202, help="Maximum heart rate achieved by the patient during exercise stress test.")
    ExerciseAngina = st.selectbox('ExerciseAngina', ('Y', 'N'), help="Presence of exercise-induced angina (Y - if present, N - if not present).")
    Oldpeak = st.slider('Old peak', -2.6, 6.2, step=0.1, help="Exercise-induced ST segment depression in the electrocardiogram.")
    ST_Slope = st.selectbox('ST Slope', ('Up', 'Flat', 'Down'), help="Slope of the ST segment in the electrocardiogram (Up - upsloping, Flat - flat).")

    data = {'Age': Age,
            'Sex': Sex,
            'ChestPainType': ChestPainType,
            'RestingBP': RestingBP,
            'Cholesterol': Cholesterol,
            'FastingBS': FastingBS,
            'RestingECG': RestingECG,
            'MaxHR': MaxHR,
            'ExerciseAngina': ExerciseAngina,
            'Oldpeak': Oldpeak,
            'ST_Slope': ST_Slope, }
    medical_data = pd.DataFrame(data, index=[0])
# input data by user
users_input = medical_data

#files
lottie_page = load_lottiefile("animations/heart-pulse.json")
lottie_prediction = load_lottiefile("animations/heart_transparent.json")

#page
st.title("Heart Disease Detector")
col1, col2 = st.columns(2)
with col2:
    st_lottie(
        lottie_page,
        speed=1,
        reverse=False,
        loop=True,
        quality="medium",
        height=550,
        width=None,
        key=None,
    )
with col1:
    st.info("This App will help you can track your risk of developing heart disease!")
    st.info("""
    Healthcare advancements have shown that preemptively predicting diseases can significantly impact patient well-being. 
    This model will scrutinize diverse health indicators, encompassing factors such as age, gender, lifestyle choices, genetic predispositions, and current health conditions. 
    Implementing such a tool holds promise for early detection, enabling timely interventions, personalized healthcare strategies, and ultimately, enhancing overall health outcomes.
                    """)

heart_data = read_data()
heart_data .dropna(inplace=True)
X = heart_data.drop(columns=['HeartDisease'])
df = pd.concat([users_input, X], axis=0)
df= pd.get_dummies(df,
                      columns=['Sex', 'ChestPainType','FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])
df = df[:1]

#load Random Forest
#Following a analysis of all algorithms, Random Forest was selected as the most effective one.
pickle_RandomForest = pickle.load(open('pickle/heart_disease_modelRF.pkl', 'rb'))
# Predictions
predictionRF = pickle_RandomForest.predict(df)
predictionRF_proba = pickle_RandomForest.predict_proba(df)


def RandomForrest():
    prediction_RF = np.array([0, 1])
    if prediction_RF[predictionRF] == 1:
        st.write("<p style='color:#E62944; font-size:2rem; text-align:center;'>You have heart disease.</p>",unsafe_allow_html=True)
    else:
        st.write("<p style='color:#0054AB; font-size:2rem; text-align:center;'> You don't have heart disease</p>",unsafe_allow_html=True)
    st_lottie(
        lottie_prediction,
        speed=1,
        reverse=False,
        loop=True,
        quality="medium",
        height=200,
        width=None,
        key=None,)
    with st.expander("More details about Random Forest"):
        ClassifierModelsHeart.info_RandomForest()
        st.write("<p style='color:#171717; text-align:center;'> Random Forest Probability <p/>", unsafe_allow_html=True )
        st.table(predictionRF_proba)


st.info("⬅ Please input your data on the sidebar")
st.markdown("""---""")
st.write("<p style='color:#171717'>Your input values are shown below:</p>",unsafe_allow_html=True)
st.table(users_input)
st.markdown("""---""")
st.header("Prediction Results")
st.write( "<p style='color:#171717; font-size: 1.35rem; text-align:center;'> Following a analysis of all algorithms, Random Forest was selected as the most effective one. </p>", unsafe_allow_html=True)
enabled = st_toggle_switch("Details about models prediction")
if enabled:
    st.write(
        "<p style='color:#525252; font-size: 1.1rem; text-align:right;'>All models created for prediction are located in the 'models' tab.</p>", unsafe_allow_html=True)

if st.button("Run Prediction"):
            progress_text = "Random Forest in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.05)
                my_bar.progress(percent_complete +1, text=progress_text)
            my_bar.empty()
            st.markdown("""---""")
            RandomForrest()





