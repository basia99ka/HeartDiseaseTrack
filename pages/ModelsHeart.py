import streamlit as st
import pandas as pd
import json
from streamlit_lottie import st_lottie
from Models import ClassifierModelsHeart
import pickle
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Models",
    page_icon="🧬",
    layout="wide"
)
#CSS streamlit style
with open ("app_style/streamlit.css") as f:
    st.markdown(f'<style> {f.read()} </style>', unsafe_allow_html=True)
#Lottiefile
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

#Loader data
def read_data():
    delimiter = ';'
    df = pd.read_csv("dataset/HeartDataset.csv", delimiter=delimiter)
    df['Oldpeak'] = df['Oldpeak'].str.replace(',', '.').astype(float)
    return df

#lottie_sider = load_lottiefile("animations/hear.json")
image = Image.open('png/anatomical-heart_1fac0.png')


#Sidebar
with st.sidebar:
    st.image(image)
    st.title("Heart Disease Track")
    st.header('Please, fill your informations to predict your heart condition')
    Age = st.slider('Age', 28, 77)
    Sex = st.selectbox('Sex', ('M', 'F'))
    ChestPainType = st.selectbox('Chest Pain Type', ('ASY', 'ATA', 'NAP', 'TA'))
    RestingBP = st.sidebar.slider('Resting Blood Pressure', 0, 200)
    Cholesterol = st.slider('Cholesterol', 0, 600)
    FastingBS = st.slider('Fasting Blood Sugar', 0, 1)
    RestingECG = st.selectbox('Resting Electrocardiogram', ('Normal', 'ST', 'LVH'))
    MaxHR = st.slider('Maximum Heart Rate', 60, 202)
    ExerciseAngina = st.selectbox('ExerciseAngina', ('Y', 'N'))
    Oldpeak = st.slider('Old peak', -2.6, 6.2, step=0.1)
    ST_Slope = st.selectbox('ST Slope', ('Up', 'Flat', 'Down'))
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
#input data by user
users_input = medical_data

#page
st.title("Heart Disease Track")
st.info(" Here you can explore other models built for prediction, see more information about the machine learning algorithms used and also evaluation of classification models.")
st.info("⬅ Please input your data on the sidebar")
st.markdown("""---""")
st.write("<p style='color:#171717'>Your input values are shown below:</p>",unsafe_allow_html=True)
st.table(users_input)

# Load models
pickle_RandomForest = pickle.load(open('pickle/heart_disease_modelRF.pkl', 'rb'))
pickle_DecisionTree = pickle.load(open('pickle/heart_disease_modelDT.pkl', 'rb'))
pickle_KNN = pickle.load(open('pickle/heart_disease_modelKNN.pkl', 'rb'))
pickle_AdaBoost = pickle.load(open('pickle/heart_disease_modelAB.pkl', 'rb'))
pickle_LogisticRegression = pickle.load(open('pickle/heart_disease_modelLR.pkl', 'rb'))

#data
heart_data = read_data()
heart_data .dropna(inplace=True)
X = heart_data.drop(columns=['HeartDisease'])
df = pd.concat([users_input, X], axis=0)
df= pd.get_dummies(df,
                      columns=['Sex', 'ChestPainType','FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])
df = df[:1]

# Predictions
predictionRF = pickle_RandomForest.predict(df)
predictionRF_proba = pickle_RandomForest.predict_proba(df)
predictionKNN = pickle_KNN.predict(df)
predictionKNN_proba = pickle_KNN.predict_proba(df)
predictionDT = pickle_DecisionTree.predict(df)
predictionDT_proba = pickle_DecisionTree.predict_proba(df)
predictionAB = pickle_AdaBoost.predict(df)
predictionAB_proba = pickle_AdaBoost.predict_proba(df)
predictionLR = pickle_LogisticRegression.predict(df)
predictionLR_proba = pickle_LogisticRegression.predict_proba(df)

#images for info
image_RF=Image.open('png/random-forest-algorithm.png')
image_DT=Image.open('png/decisionTree.png')
image_LR=Image.open('png/LogReg.png')
image_AB= Image.open('png/Boosting.png')
image_KNN = Image.open('png/KNN_.png')

def RandomForrest():
    st.header('Random Forrest Prediction')
    prediction_RF = np.array([0, 1])
    if prediction_RF[predictionRF] == 1:
        st.write("<p style='color:#E62944; font-size:2rem; text-align:center;'>You have heart disease.</p>",unsafe_allow_html=True)
    else:
        st.write("<p style='color:#0054AB; font-size:2rem; text-align:center;'> You don't have heart disease</p>",unsafe_allow_html=True)
    st.markdown("""---""")
    col1, col2 = st.columns(2)
    with col1:
        st.info("Random Forest is a popular machine learning algorithm that belongs to the supervised learning technique. It can be used for both Classification and Regression problems in ML. It is based on the concept of ensemble learning, which is a process of combining multiple classifiers to solve a complex problem and to improve the performance of the model.")
    with col2:
        st.image(image_RF)
    ClassifierModelsHeart.info_RandomForest()
    st.write("<p style='color:#171717; text-align:center;'> Random Forest Probability <p/>", unsafe_allow_html=True )
    st.table(predictionRF_proba)


def DecisionTree():
    st.header('Decision Tree Prediction')
    prediction_DT = np.array([0, 1])
    if prediction_DT[predictionDT] == 1:
        st.write("<p style='color:#E62944; font-size:2rem; text-align:center;'>You have heart disease.</p>",unsafe_allow_html=True)
    else:
        st.write("<p style='color:#0054AB; font-size:2rem; text-align:center;'> You don't have heart disease</p>",unsafe_allow_html=True)
    st.markdown("""---""")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        Decision tree is one of the most powerful tools of supervised learning algorithms used for both classification and regression tasks. It builds a flowchart-like tree structure where each internal node denotes a test on an attribute, each branch represents an outcome of the test, and each leaf node (terminal node) holds a class label. It is constructed by recursively splitting the training data into subsets based on the values of the attributes until a stopping criterion is met, such as the maximum depth of the tree or the minimum number of samples required to split a node.
        During training, the Decision Tree algorithm selects the best attribute to split the data based on a metric such as entropy or Gini impurity, which measures the level of impurity or randomness in the subsets. The goal is to find the attribute that maximizes the information gain or the reduction in impurity after the split.
        """)
    with col2:
        st.image(image_DT)
    ClassifierModelsHeart.info_DecisionTree()
    st.write("<p style='color:#171717; text-align:center;'> Decision Tree Probability <p/>", unsafe_allow_html=True)
    st.table(predictionDT_proba)

def KNN():
    st.header('Prediction KNN')
    prediction_KNN = np.array([0, 1])
    if prediction_KNN[predictionKNN] == 1:
        st.write("<p style='color:#E62944; font-size:2rem; text-align:center;'>You have heart disease.</p>",
                 unsafe_allow_html=True)
    else:
        st.write("<p style='color:#0054AB; font-size:2rem; text-align:center;'> You don't have heart disease</p>",unsafe_allow_html=True)
    
    st.markdown("""---""")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""The K-Nearest Neighbors (KNN) algorithm is a supervised machine learning method employed to tackle classification and regression problems.
     In both cases, the input consists of the k closest training examples in a data set. 
     For classification problems, a class label is assigned on the basis of a majority vote—i.e. the label that is most frequently represented around a given data point is used. 
     Regression problems use a similar concept as classification problem, but in this case, the average the k nearest neighbors is taken to make a prediction about a classification.""")
    with col2:
        st.image(image_KNN)
    ClassifierModelsHeart.info_KNN()
    st.write("<p style='color:#171717; text-align:center;'> KNN Probability <p/>", unsafe_allow_html=True)
    st.table(predictionKNN_proba)

def LogisticRegression():
    st.header('Logistic Regression Prediction')
    prediction_LR = np.array([0, 1])
    if prediction_LR[predictionLR] == 1:
        st.write("<p style='color:#E62944; font-size:2rem; text-align:center;'>You have heart disease.</p>",unsafe_allow_html=True)
    else:
        st.write("<p style='color:#0054AB; font-size:2rem; text-align:center;'> You don't have heart disease</p>",unsafe_allow_html=True)
    
    st.markdown("""---""")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""Logistic regression is a supervised machine learning algorithm used for classification tasks where the goal is to predict the probability that an instance belongs to a given class or not. 
        Logistic regression is a statistical algorithm which analyze the relationship between two data factors.
        This algorithm predicts the output of a categorical dependent variable. It can be either Yes or No, 0 or 1, true or False, etc. but instead of giving the exact value as 0 and 1, 
        it gives the probabilistic values which lie between 0 and 1
        The logistic regression model transforms the linear regression function continuous value output into categorical value output using a sigmoid function, which maps any real-valued set of independent variables input into a value between 0 and 1. This function is known as the logistic function.""")
    with col2:
        st.image(image_LR)

    ClassifierModelsHeart.info_LogisticRegression()
    st.write("<p style='color:#171717; text-align:center;'> Logistic Regression Probability <p/>", unsafe_allow_html=True)
    st.table(predictionLR_proba)


def AdaBoost():
    st.header('AdaBoost Prediction')
    prediction_AB = np.array([0, 1])
    if prediction_AB[predictionAB] == 1:
        st.write("<p style='color:#E62944; font-size:2rem; text-align:center;'>You have heart disease.</p>",unsafe_allow_html=True)
    else:
        st.write("<p style='color:#0054AB; font-size:2rem; text-align:center;'> You don't have heart disease</p>",unsafe_allow_html=True)
    st.markdown("""---""")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**AdaBoost**, short for Adaptive Boosting, is an ensemble machine learning algorithm that can be used in a wide variety of classification and regression tasks. It is a supervised learning algorithm that is used to classify data by combining multiple weak or base learners (e.g., decision trees) into a strong learner. AdaBoost works by weighting the instances in the training dataset based on the accuracy of previous classifications. AdaBoost works by weighting the instances in the training dataset based on the accuracy of previous classifications.")
    with col2:
        st.image(image_AB)
    ClassifierModelsHeart.info_AdaBoost()
    st.write("<p style='color:#171717; text-align:center;'> Ada Boost Probability <p/>", unsafe_allow_html=True)
    st.table(predictionAB_proba)


# Select the model
selected_model = st.selectbox(":gray[Select Model]",["Random Forrest", "K-Nearest Neighbors (KNN)","Decision Tree",
                                              "AdaBoost","Logistic Regression"], index=None, placeholder="Choose Model")
if selected_model == "K-Nearest Neighbors (KNN)":
    KNN()
elif selected_model == "Random Forrest":
    RandomForrest()
elif selected_model == "Decision Tree":
    DecisionTree()
elif selected_model == "AdaBoost":
    AdaBoost()
elif selected_model == "Logistic Regression":
    LogisticRegression()
