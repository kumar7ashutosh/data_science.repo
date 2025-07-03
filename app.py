import streamlit as st,pandas as pd,numpy as np,pickle
with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
with open('ridge.pkl', 'rb') as f:
        model = pickle.load(f)
        
st.set_page_config(page_title="Algerian Forest Fire FWI Predictor", layout="centered")

st.title("🔥 Algerian Forest Fire FWI Predictor 🔥")
st.markdown("Enter the weather conditions below to predict the Fire Weather Index (FWI).")

st.header("Weather Conditions Input")

temperature = st.slider("Temperature (°C)", min_value=25, max_value=42, value=35)
rh = st.slider("Relative Humidity (%)", min_value=20, max_value=90, value=60)
ws = st.slider("Wind Speed (km/h)", min_value=7, max_value=35, value=15)
rain = st.number_input("Rain (mm)", min_value=0.0, max_value=17.0, value=0.0, step=0.1)
ffmc = st.number_input("Fine Fuel Moisture Code (FFMC)", min_value=20.0, max_value=95.0, value=85.0, step=0.1)
dmc = st.number_input("Duff Moisture Code (DMC)", min_value=1.0, max_value=60.0, value=15.0, step=0.1)
dc = st.number_input("Drought Code (DC)", min_value=7.0, max_value=250.0, value=60.0, step=0.1)
isi = st.number_input("Initial Spread Index (ISI)", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
bui = st.number_input("Buildup Index (BUI)", min_value=5.0, max_value=70.0, value=20.0, step=0.1)
fire_class_input = st.selectbox("Fire Class (0: Not Fire, 1: Fire)", options=[0, 1])

input_data = pd.DataFrame([[
    temperature, rh, ws, rain, ffmc, dmc, dc, isi, bui, fire_class_input
]], columns=['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'Classes'])
input_data.drop(columns=['BUI','DC'],axis=1,inplace=True)
st.header("Prediction")
if st.button("Predict FWI"):
    scaled_input_data = scaler.transform(input_data)

    prediction = model.predict(scaled_input_data)[0]

    st.subheader("Predicted Fire Weather Index (FWI):")
    st.success(f"**{prediction:.2f}**")
