import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model and transformers
with open('ai_model10.pkl', 'rb') as file:
    model = pickle.load(file)

with open('level_of_government_encoder.pkl', 'rb') as file:
    level_of_government_encoder = pickle.load(file)

with open('one_hot_encoder.pkl', 'rb') as file:
    one_hot_encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the revenue category mapping
revenue_category_mapping = {
    1: 'Total tax and non-tax revenue', 
    2: 'Total tax revenue',
    3: '1000 Taxes on income, profits and capital gains',
    # Add more categories here...
}

# Reverse the mapping for displaying in the selectbox
revenue_category_options = {v: k for k, v in revenue_category_mapping.items()}

# Define the app
st.title("Tax Revenue Prediction App")

# Input fields
level_of_government_options = level_of_government_encoder.classes_
level_of_government = st.selectbox("Select Level of Government", level_of_government_options)
revenue_category = st.selectbox("Select Revenue Category", list(revenue_category_options.keys()))
year = st.number_input("Enter Year", min_value=2001, max_value=2024, step=1)

# Predict button
if st.button("Predict"):
    # Encode inputs
    level_of_government_encoded = level_of_government_encoder.transform([level_of_government])[0]
    revenue_category_encoded = revenue_category_options[revenue_category]

    # Create input dataframe
    input_data = pd.DataFrame({
        'Level of government': [level_of_government_encoded],
        'Revenue category': [revenue_category_encoded],
        'Year': [year]
    })

    # One-hot encode categorical features
    encoded_features = one_hot_encoder.transform(input_data[['Level of government', 'Revenue category']])
    
    # Scale the 'Year' column
    input_data[['Year']] = scaler.transform(input_data[['Year']])
    
    # Combine encoded features with scaled year
    input_features = np.concatenate((input_data[['Year']], encoded_features), axis=1)

    # Make prediction
    prediction = model.predict(input_features)

    # Display the prediction
    st.write(f"Predicted Revenue: {prediction[0]:,.2f} Millions")
