import pandas as pd
import streamlit as st
from models.model import svm_cardio_predict

import matplotlib.pyplot as plt
import seaborn as sns
import requests

st.header('Cardiovascular Disease Prediction')
st.subheader('Using SVM')


features, scaler, model, Y_pred, cr, cm = svm_cardio_predict()

st.sidebar.header(
    'Cardio Features'
)

age = st.sidebar.slider(
    'Age',
    max_value = 70,
    min_value = 26,
    value = 30,
    step = 5
)

gender_dict = {1: 'Female', 2: 'Male'}

selected_gender = st.sidebar.selectbox(
   'Gender',
   options = list(gender_dict.keys()),
   format_func = lambda x : gender_dict.get(x)
)
gender = selected_gender

# shift, alt, downarrow to copy down
height = st.sidebar.slider(
    'Height',
    max_value = 200,
    min_value = 136,
    value = 145,
    step = 1
)

weight = st.sidebar.slider(
    'Weight',
    max_value = 120,
    min_value = 35,
    value = 60,
    step = 1
)

ap_hi = st.sidebar.slider(
    'Systolic Pressure',
    max_value = 200,
    min_value = 90,
    value = 120,
    step = 1
)

ap_lo = st.sidebar.slider(
    'Dysystolic Pressure',
    max_value = 100,
    min_value = 50,
    value = 80,
    step = 1
)

cholesterol_dict = {
    1: 'Low Cholesterol',
    2: 'Mild Cholesterol',
    3: 'High Cholesterol'
}
cholesterol = st.sidebar.selectbox(
    'Cholesterol',
    options = list(cholesterol_dict.keys()),
    format_func = lambda x : cholesterol_dict.get(x)
)

gluc_dict = {
    1: 'Low Glucose',
    2: 'Mild Glucose',
    3: 'High Glucose'
}
gluc = st.sidebar.selectbox(
    'Glucose',
    options = list(gluc_dict.keys()),
    format_func = lambda x : gluc_dict.get(x)
)

smoke_dict = {0: 'Does not Smoke', 1: 'Does Smoke'}
smoke = st.sidebar.radio(
    'Smoke',
    options = list(smoke_dict.keys()),
    format_func = lambda x : smoke_dict.get(x)
)

alco_dict = {0: 'Does not Drink Alcohol',
              1: 'Does Drink Alcohol'}
alco = st.sidebar.selectbox(
    'Alcohol',
    options = list(alco_dict.keys()),
    format_func = lambda x : alco_dict.get(x)
)
active_dict = {0: 'Does not do PA',
              1: 'Does do PA'}
active = st.sidebar.selectbox(
    'Physical Activities (PA)',
    options = list(active_dict.keys()),
    format_func = lambda x : active_dict.get(x)
)

if st.button('Predict Cardio'):
    input_data = pd.DataFrame([[
        age, gender, height, weight, ap_hi, ap_lo,
        cholesterol, gluc, smoke, alco, active
    ]], columns = features)

   # Data scaling
    input_scaler = scaler.transform(input_data)

   # Predict using model
    prediction = model.predict(input_scaler)[0]

   # Show Answer
    if prediction == 0:
        st.write('Likely not to have cardiovascular disease.')
        st.success('No cardiovascular disease found. ')
    else:
        st.write('Likely to have cardiovascular disease.')
        st.warning('Cardiovascular disease found. ')


# Visualization
st.subheader('Visualization')

fig, ax = plt.subplots(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=['Predicted Healthy[0]', 'Predicted UnHealthy[1]'],
            yticklabels=['Actual Healthy[0]', 'Actual UnHealthy[1]'])
plt.title('Actual Cardio vs. Predicted Cardio')
st.pyplot(fig)
