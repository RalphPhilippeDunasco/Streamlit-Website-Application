import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('milk_quality_model.h5')
    return model
model = load_model()

data = pd.read_csv("milknew.csv")

X = data.drop(columns=['Grade'])
y = data['Grade']

scaler = StandardScaler()
scaler.fit(X)

label_encoder = LabelEncoder()
data['Grade'] = label_encoder.fit_transform(data['Grade'])

page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
    background: rgba(0, 0, 0, 0) url("https://media.istockphoto.com/id/1138321469/video/pouring-milk-into-a-glass-on-black-background.jpg?s=640x640&k=20&c=tA7lTUAXeip3ytOLxeHA7sQojwPQuqa_cxZFFJVMwqM=");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("Milk Quality Prediction")
st.text("Predicting the Quality of Milk Based on 7 Independent Variables")
st.text("Results: Low (Bad), Medium (Moderate), High (Good)")

pH = st.select_slider('pH', options=[round(i, 1) for i in np.arange(3, 9.6, 0.1)])
Temprature = st.slider('Temprature (°C)', 34, 90, step=1)
Taste = st.selectbox('Taste (0: Bad | 1: Good)', options=[0, 1])
Odor = st.selectbox('Odor (0: Bad | 1: Good)', options=[0, 1])
Fat = st.selectbox('Fat (0: Low | 1: High)', options=[0, 1])
Turbidity = st.selectbox('Turbidity (0: Low | 1: High)', options=[0, 1])
Colour = st.slider('Colour', 240, 255, step=1)

user_input = {
    'pH': [pH],
    'Temprature': [Temprature],
    'Taste': [Taste],
    'Odor': [Odor],
    'Fat ': [Fat],
    'Turbidity': [Turbidity],
    'Colour': [Colour]
}

def preprocess_input(user_input):
    input_df = pd.DataFrame(user_input)
    input_scaled = scaler.transform(input_df)
    return input_scaled

def make_predictions(input_data):
    processed_input = preprocess_input(input_data)
    predictions = model.predict(processed_input)
    predicted_class = label_encoder.inverse_transform(predictions.argmax(axis=-1))
    return predicted_class[0]

if st.button('Predict'):
    predicted_grade = make_predictions(user_input)
    st.success(f"The predicted grade of milk is of **{predicted_grade}** quality")
