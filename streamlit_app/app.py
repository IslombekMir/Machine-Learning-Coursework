import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'xgb_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler_runtime.pkl')
MLB_PATH = os.path.join(BASE_DIR, 'mlb_genres.pkl')

# --- Load model and preprocessing objects ---
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

with open(MLB_PATH, 'rb') as f:
    mlb = pickle.load(f)

# --- Streamlit app ---
st.title("Movie Rating Prediction")

# Inputs
runtime = st.number_input("Runtime (minutes)", min_value=0)
is_adult = st.selectbox("Adult Movie?", [0, 1])
selected_genres = st.multiselect("Select Genres", options=mlb.classes_)

# --- Preprocess inputs ---
# Scale runtime
runtime_scaled = scaler.transform([[runtime]])[0][0]

# Encode genres
genres_encoded = mlb.transform([selected_genres])
genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)

# Create input DataFrame
input_data = pd.DataFrame({
    'runtimeMinutes_scaled': [runtime_scaled],
    'isAdult': [is_adult]
})
input_data = pd.concat([input_data, genres_df], axis=1)

# Ensure all model columns exist and in correct order
for col in model.get_booster().feature_names:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[model.get_booster().feature_names]

# --- Prediction ---
if st.button("Predict Rating"):
    pred = model.predict(input_data)
    st.write(f"Predicted Rating: {pred[0]:.2f}")