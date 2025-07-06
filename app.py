#Import all the neccesary libraries
import pandas as pd
import numpy as np
import joblib
import pickle
import streamlit as st

#Loading the model and structure
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")

#Creating an User interface
st.title("Water Pollutant Predictor")
st.write("Predict the water pollutants based on Year and Station ID")

#User inputs
year_input = st.number_input("Enter Year",min_value=2000,max_value=2100, value=2004)
station_id = st.text_input("Enter Station ID", value='1')

#To encode adn predict 
if st.button('Predict'):
    if not station_id:
        st.warning('Please enter the station ID')
    else:
        #Prepare the input
        input_df = pd.DataFrame({'year': [year_input], 'id':[station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])

        #Align with model cols
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        #Predict 
        predicted_pollutants = model.predict(input_encoded)[0]
        pollutants = ['O2', 'NO3', 'SO4', 'PO4', 'CL']

        st.subheader(f"Predicted pollutant levels for the station '{station_id}' in {year_input}:")
        predicted_values = {}
        for p, val in zip(pollutants, predicted_pollutants):
            st.write(f'{p}:{val:2f}')
