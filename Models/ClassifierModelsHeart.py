import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
import pickle

def read_data():
    delimiter = ';'
    df = pd.read_csv("dataset/HeartDataset.csv", delimiter=delimiter)
    df['Oldpeak'] = df['Oldpeak'].str.replace(',', '.').astype(float)
    return df

df = read_data()
# Filling in missing data
df.dropna(inplace=True)
heart = df.copy()

# One-hot encoding
heart= pd.get_dummies(heart,
                      columns=['Sex', 'ChestPainType','FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])
# Prepare X and y
X = heart.drop('HeartDisease', axis=1)
Y = heart['HeartDisease']

# Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# ---- RANDOM FORREST ----
# Random Forest classifier
classifier_RF = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)
classifier_RF.fit(X_train, y_train)
prediction_RF = classifier_RF.predict(X_test)
matrixC_RF = confusion_matrix(y_test, prediction_RF)
recall_RF = recall_score(y_test, prediction_RF)
report_RF = classification_report(y_test, prediction_RF)
report_RFdict = classification_report(y_test, prediction_RF, output_dict=True)
report_RF_df = pd.DataFrame(report_RFdict)
importance_RF = classifier_RF.feature_importances_
importance_data_RF = pd.DataFrame({'Feature': X.columns, 'Importance': importance_RF})
importance_data_RF = importance_data_RF.sort_values(by='Importance', ascending=False)

def info_RandomForest():
    st.header("Evaluating a classification model - Random Forest")
    st.write("<p style='color:#171717; text-align:center;'>Random Forest Classifier Report  </p>", unsafe_allow_html=True)
    st.table(report_RF_df)
    col1,col2 = st.columns(2)
    with col1:
        st.write("<p style='color:#171717; text-align:center;'> Feature Importance Plot </p>",
                 unsafe_allow_html=True)
        fig = px.bar(importance_data_RF, x='Importance', y='Feature', orientation='h')
        fig.update_yaxes(tickangle=90)
        fig.update_layout(
           yaxis=dict(tickangle=0),
           xaxis=dict(title='Importance'),
           plot_bgcolor='#121820',
           paper_bgcolor='#121820',
           font=dict(family='Arial', size=15, color='white'),
           height=600,
           width=800)
        fig.update_traces(marker_color='#3A86FF',
                         marker_line_color='#20365B',
                         marker_line_width=1.5,
                         opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.write("<p style='color:#171717; text-align:center;'> Confusion Matrix </p>",
                 unsafe_allow_html=True)
        chart_RF = px.imshow(matrixC_RF,
                     labels=dict(x="Predicted Label", y="True Label", color="Count"),
                     x=['No Disease', 'Disease'],
                     y=['No Disease', 'Disease'],
                     )
        chart_RF.update_layout(
           coloraxis_colorbar=dict(title="Count"),
           plot_bgcolor = '#121820',
           paper_bgcolor = '#121820',
           font = dict(family='Arial', size=15, color='white'),)
        chart_RF.update_xaxes(side="top")
        st.plotly_chart(chart_RF, use_container_width=True)

#---- KNN Algorithm ----

classifier_KNN = KNeighborsClassifier()
classifier_KNN.fit(X_train, y_train)
prediction_KNN = classifier_KNN.predict(X_test)
matrixC_KNN = confusion_matrix(y_test, prediction_KNN)
accuracy_KNN= accuracy_score(y_test, prediction_KNN)
recall_KNN = recall_score(y_test, prediction_KNN)
report_KNN = classification_report(y_test, prediction_KNN)
report_KNNdict = classification_report(y_test, prediction_KNN, output_dict=True)
report_KNN_df = pd.DataFrame(report_KNNdict)

def info_KNN():
    st.header("Evaluating a classification model - KNN")
    col1, col2 = st.columns(2)
    with col1:
        st.write("<p style='color:#171717; text-align:center;'>KNN Classifier Report  </p>",
                 unsafe_allow_html=True)
        st.table(report_KNN_df)
    with col2:
        st.write("<p style='color:#171717; text-align:center;'> Confusion Matrix </p>",
                 unsafe_allow_html=True)
        chart_KNN = px.imshow(matrixC_KNN,
                             labels=dict(x="Predicted Label", y="True Label", color="Count"),
                             x=['No Disease', 'Disease'],
                             y=['No Disease', 'Disease'],
                             )
        chart_KNN.update_layout(
            coloraxis_colorbar=dict(title="Count"),
            plot_bgcolor='#121820',
            paper_bgcolor='#121820',
            font=dict(family='Arial', size=15, color='white'), )
        chart_KNN.update_xaxes(side="top")
        st.plotly_chart(chart_KNN, use_container_width=True)


#---- Decision Tree ----
classifier_DT = DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=42)
classifier_DT.fit(X_train, y_train)
prediction_DT = classifier_DT.predict(X_test)
matrixC_DT = confusion_matrix(y_test, prediction_DT)
accuracy_DT= accuracy_score(y_test, prediction_DT)
report_DT= classification_report(y_test, prediction_DT)
report_DTdict = classification_report(y_test, prediction_DT, output_dict=True)
report_DT_df = pd.DataFrame(report_DTdict)
recall_DT = recall_score(y_test, prediction_DT)
importance_DT = classifier_DT.feature_importances_
importance_data_DT = pd.DataFrame({'Feature': X.columns, 'Importance': importance_DT})
importance_data_DT = importance_data_DT.sort_values(by='Importance', ascending=False)


def info_DecisionTree():
    st.header("Evaluating a classification model - Decision Tree")
    st.write("<p style='color:#171717; text-align:center;'>Decision Tree Classifier Report  </p>", unsafe_allow_html=True)
    st.table(report_DT_df)
    col1,col2 = st.columns(2)
    with col1:
        st.write("<p style='color:#171717; text-align:center;'> Feature Importance Plot </p>",
                 unsafe_allow_html=True)
        fig_DT = px.bar(importance_data_DT, x='Importance', y='Feature', orientation='h')
        fig_DT.update_yaxes(tickangle=90)
        fig_DT.update_layout(
           yaxis=dict(tickangle=0),
           xaxis=dict(title='Importance'),
           plot_bgcolor='#121820',
           paper_bgcolor='#121820',
           font=dict(family='Arial', size=15, color='white'),
           height=600,
           width=800)
        fig_DT.update_traces(marker_color='#3A86FF',
                         marker_line_color='#20365B',
                         marker_line_width=1.5,
                         opacity=0.6)
        st.plotly_chart(fig_DT, use_container_width=True)
    with col2:
        st.write("<p style='color:#171717; text-align:center;'> Confusion Matrix </p>",
                 unsafe_allow_html=True)
        chart_DT = px.imshow(matrixC_DT,
                     labels=dict(x="Predicted Label", y="True Label", color="Count"),
                     x=['No Disease', 'Disease'],
                     y=['No Disease', 'Disease'],
                     )
        chart_DT.update_layout(
           coloraxis_colorbar=dict(title="Count"),
           plot_bgcolor = '#121820',
           paper_bgcolor = '#121820',
           font = dict(family='Arial', size=15, color='white'),)
        chart_DT.update_xaxes(side="top")
        st.plotly_chart(chart_DT, use_container_width=True)


#--- ADABOOST ---
classifier_AB = AdaBoostClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
classifier_AB.fit(X_train, y_train)
prediction_AB = classifier_AB.predict(X_test)
matrixC_AB = confusion_matrix(y_test, prediction_AB)
accuracy_AB= accuracy_score(y_test, prediction_AB)
report_AB= classification_report(y_test, prediction_AB)
report_ABdict = classification_report(y_test, prediction_AB, output_dict=True)
report_AB_df = pd.DataFrame(report_ABdict)
recall_AB = recall_score(y_test, prediction_AB)
importance_AB = classifier_AB.feature_importances_
importance_data_AB = pd.DataFrame({'Feature': X.columns, 'Importance': importance_AB})
importance_data_AB = importance_data_AB.sort_values(by='Importance', ascending=False)

def info_AdaBoost():
    st.header("Evaluating a classification model - AdaBoost")
    st.write("<p style='color:#171717; text-align:center;'>AdaBoost Classifier Report  </p>", unsafe_allow_html=True)
    st.table(report_AB_df)
    col1,col2 = st.columns(2)
    with col1:
        st.write("<p style='color:#171717; text-align:center;'> Feature Importance Plot </p>",
                 unsafe_allow_html=True)
        fig_AB = px.bar(importance_data_AB, x='Importance', y='Feature', orientation='h')
        fig_AB.update_yaxes(tickangle=90)
        fig_AB.update_layout(
           yaxis=dict(tickangle=0),
           xaxis=dict(title='Importance'),
           plot_bgcolor='#121820',
           paper_bgcolor='#121820',
           font=dict(family='Arial', size=15, color='white'),
           height=600,
           width=800)
        fig_AB.update_traces(marker_color='#3A86FF',
                         marker_line_color='#20365B',
                         marker_line_width=1.5,
                         opacity=0.6)
        st.plotly_chart(fig_AB, use_container_width=True)
    with col2:
        st.write("<p style='color:#171717; text-align:center;'> Confusion Matrix </p>",
                 unsafe_allow_html=True)
        chart_AB = px.imshow(matrixC_AB,
                     labels=dict(x="Predicted Label", y="True Label", color="Count"),
                     x=['No Disease', 'Disease'],
                     y=['No Disease', 'Disease'],
                     )
        chart_AB.update_layout(
           coloraxis_colorbar=dict(title="Count"),
           plot_bgcolor = '#121820',
           paper_bgcolor = '#121820',
           font = dict(family='Arial', size=15, color='white'),)
        chart_AB.update_xaxes(side="top")
        st.plotly_chart(chart_AB, use_container_width=True)


#----LogisticRegression----

classifier_LR = LogisticRegression()
classifier_LR.fit(X_train, y_train)
prediction_LR = classifier_LR.predict(X_test)
matrixC_LR = confusion_matrix(y_test, prediction_LR)
accuracy_LR = accuracy_score(y_test, prediction_LR)
report_LR = classification_report(y_test, prediction_LR)
report_LRdict = classification_report(y_test, prediction_LR, output_dict=True)
report_LR_df = pd.DataFrame(report_LRdict)
recall_LR = recall_score(y_test, prediction_LR)

def info_LogisticRegression():
    st.header("Evaluating a classification model - Logistic Regression")
    col1,col2 = st.columns(2)
    with col1:
        st.write("<p style='color:#171717; text-align:center;'>Logistic Regression Classifier Report  </p>",
                 unsafe_allow_html=True)
        st.table(report_LR_df)
    with col2:
        st.write("<p style='color:#171717; text-align:center;'> Confusion Matrix </p>",
                 unsafe_allow_html=True)
        chart_LR = px.imshow(matrixC_LR,
                     labels=dict(x="Predicted Label", y="True Label", color="Count"),
                     x=['No Disease', 'Disease'],
                     y=['No Disease', 'Disease'],
                     )
        chart_LR.update_layout(
           coloraxis_colorbar=dict(title="Count"),
           plot_bgcolor = '#121820',
           paper_bgcolor = '#121820',
           font = dict(family='Arial', size=15, color='white'),)
        chart_LR.update_xaxes(side="top")
        st.plotly_chart(chart_LR, use_container_width=True)



# Select the best model for prediction
models = {
    'Random Forest': report_RF,
    'Decision Tree': report_DT,
    'K-Nearest Neighbors (KNN)': report_KNN,
    'Logistic Regression': report_LR,
    'ADABOOST': report_AB,
}
best_model = max(models, key=models.get)


# save models - pickle
pickle.dump(classifier_KNN, open('pickle/heart_disease_modelKNN.pkl', 'wb'))
pickle.dump(classifier_DT, open('pickle/heart_disease_modelDT.pkl', 'wb'))
pickle.dump(classifier_AB, open('pickle/heart_disease_modelAB.pkl', 'wb'))
pickle.dump(classifier_LR, open('pickle/heart_disease_modelLR.pkl', 'wb'))
pickle.dump(classifier_RF, open('pickle/heart_disease_modelRF.pkl', 'wb'))
