import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Page configuration
st.set_page_config(
    page_title='Customer Churn Predictor',
    page_icon='💼',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Load trained model and encoders/scaler
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('model.h5')
    with open('onehot_encoder_geo.pkl','rb') as f:
        geo_enc = pickle.load(f)
    with open('label_encoder_gender.pkl','rb') as f:
        gender_enc = pickle.load(f)
    with open('scaler.pkl','rb') as f:
        scaler = pickle.load(f)
    return model, geo_enc, gender_enc, scaler

model, geo_enc, gender_enc, scaler = load_assets()

# Sidebar for inputs
st.sidebar.header('Customer Features')
geography = st.sidebar.selectbox('Geography', geo_enc.categories_[0])
gender = st.sidebar.selectbox('Gender', gender_enc.classes_)
age = st.sidebar.slider('Age', 18, 92, 30)
balance = st.sidebar.number_input('Balance', min_value=0.0, format="%.2f")
credit_score = st.sidebar.number_input('Credit Score', min_value=300, max_value=850, value=600)
estimated_salary = st.sidebar.number_input('Estimated Salary', min_value=0.0, format="%.2f")
tenure = st.sidebar.slider('Tenure (years)', 0, 10, 3)
num_of_products = st.sidebar.slider('Number of Products', 1, 4, 1)
has_cr_card = st.sidebar.selectbox('Has Credit Card', (0, 1))
is_active_member = st.sidebar.selectbox('Is Active Member', (0, 1))

# Main content
st.markdown("## 💡 Customer Churn Prediction")

# Prepare input
input_df = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_enc.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})
# Geography OHE
geo_ohe = geo_enc.transform([[geography]]).toarray()
geo_df = pd.DataFrame(geo_ohe, columns=geo_enc.get_feature_names_out(['Geography']))
input_df = pd.concat([input_df, geo_df], axis=1)

# Scale
data_scaled = scaler.transform(input_df)

# Predict
if st.sidebar.button('Predict Churn'):
    proba = float(model.predict(data_scaled)[0][0])
    churn = proba > 0.5

    # Display results in columns
    col1, col2 = st.columns([2, 1])
    with col1:
        st.metric(label="Churn Probability", value=f"{proba:.2%}", delta=None)
        st.progress(proba)
    with col2:
        if churn:
            st.error('🚨 Likely to Churn')
        else:
            st.success('✅ Unlikely to Churn')

    # Show raw probabilities
    with st.expander('See raw prediction'): 
        st.write({'churn_probability': proba})

