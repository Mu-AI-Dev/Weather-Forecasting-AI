import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Advanced Weather Forecaster",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. LOAD MODELS & ASSETS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        clf = joblib.load('models/model_classifier.pkl')
        reg = joblib.load('models/model_regressor.pkl')
        scaler_clf = joblib.load('models/scaler_classifier.pkl')
        scaler_reg = joblib.load('models/scaler_regressor.pkl')
        features_clf = joblib.load('models/features_clf_order.pkl')
        # If you saved regressor features, load them. Otherwise, we assume they are a subset or similar.
        # For safety, I'll assume you might use the same features or defined them manually.
        # features_reg = joblib.load('models/features_reg_order.pkl') 
        return clf, reg, scaler_clf, scaler_reg, features_clf
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}")
        return None, None, None, None, None

clf, reg, scaler_clf, scaler_reg, features_clf = load_artifacts()

# -----------------------------------------------------------------------------
# 3. UTILITY FUNCTIONS (FEATURE ENGINEERING)
# -----------------------------------------------------------------------------
def preprocess_input(user_input, feature_order, scaler, is_regression=False):
    """
    Transforms raw user input dictionary into the exact format required by the model.
    """
    # 1. Convert dict to DataFrame
    df = pd.DataFrame([user_input])
    
    # 2. Derived Features (Math Operations)
    df['TempRange'] = df['MaxTemp'] - df['MinTemp']
    df['TempChange'] = df['Temp3pm'] - df['Temp9am']
    df['HumidityChange'] = df['Humidity3pm'] - df['Humidity9am']
    df['PressureChange'] = df['Pressure3pm'] - df['Pressure9am']
    df['WindSpeedChange'] = df['WindSpeed3pm'] - df['WindSpeed9am']
    df['CloudChange'] = df['Cloud3pm'] - df['Cloud9am']
    
    df['AvgTemp'] = (df['Temp9am'] + df['Temp3pm']) / 2
    df['AvgHumidity'] = (df['Humidity9am'] + df['Humidity3pm']) / 2
    df['AvgPressure'] = (df['Pressure9am'] + df['Pressure3pm']) / 2
    df['AvgWindSpeed'] = (df['WindSpeed9am'] + df['WindSpeed3pm']) / 2
    df['AvgCloud'] = (df['Cloud9am'] + df['Cloud3pm']) / 2
    
    # 3. Wind Direction Encoding (Cyclical)
    wind_dir_mapping = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
        'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
        'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
        'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }
    
    wind_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
    for col in wind_cols:
        val = df.iloc[0][col]
        angle = wind_dir_mapping.get(val, 0) # Default to 0 (N) if error
        df[f'{col}_sin'] = np.sin(np.radians(angle))
        df[f'{col}_cos'] = np.cos(np.radians(angle))

    # 4. Binary Encoding
    df['RainToday_encoded'] = 1 if df.iloc[0]['RainToday'] == 'Yes' else 0
    
    # 5. Select & Order Columns
    # Ensure we only pick the columns that the model expects, in order
    try:
        df_final = df[feature_order]
    except KeyError as e:
        st.error(f"Missing column in processing: {e}")
        return None
        
    # 6. Scale
    df_scaled = scaler.transform(df_final)
    
    return df_scaled

# -----------------------------------------------------------------------------
# 4. SIDEBAR - USER INPUTS
# -----------------------------------------------------------------------------
st.sidebar.title("🛠️ Input Parameters")
st.sidebar.markdown("Adjust the weather conditions below:")

with st.sidebar.form("weather_form"):
    st.subheader("🌡️ Temperature & Humidity")
    col1, col2 = st.columns(2)
    with col1:
        MinTemp = st.number_input("Min Temp (°C)", value=15.0)
        Temp9am = st.number_input("Temp 9am (°C)", value=18.0)
        Humidity9am = st.slider("Humidity 9am (%)", 0, 100, 60)
    with col2:
        MaxTemp = st.number_input("Max Temp (°C)", value=25.0)
        Temp3pm = st.number_input("Temp 3pm (°C)", value=22.0)
        Humidity3pm = st.slider("Humidity 3pm (%)", 0, 100, 50)
        
    st.subheader("💨 Wind Conditions")
    col3, col4 = st.columns(2)
    with col3:
        WindGustDir = st.selectbox("Gust Direction", ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'])
        WindDir9am = st.selectbox("Wind Dir 9am", ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'])
        WindDir3pm = st.selectbox("Wind Dir 3pm", ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'])
    with col4:
        WindGustSpeed = st.slider("Gust Speed (km/h)", 0, 150, 40)
        WindSpeed9am = st.slider("Speed 9am (km/h)", 0, 100, 15)
        WindSpeed3pm = st.slider("Speed 3pm (km/h)", 0, 100, 20)

    st.subheader("☁️ Other Factors")
    col5, col6 = st.columns(2)
    with col5:
        Pressure9am = st.number_input("Pressure 9am (hPa)", value=1015.0)
        Cloud9am = st.slider("Cloud 9am (oktas)", 0, 9, 4)
        Evaporation = st.number_input("Evaporation (mm)", value=5.0)
        RainToday = st.selectbox("Did it rain today?", ["No", "Yes"])
    with col6:
        Pressure3pm = st.number_input("Pressure 3pm (hPa)", value=1012.0)
        Cloud3pm = st.slider("Cloud 3pm (oktas)", 0, 9, 4)
        Sunshine = st.number_input("Sunshine (hours)", value=8.0)
        Rainfall = st.number_input("Rainfall Today (mm)", value=0.0)

    submitted = st.form_submit_button("🚀 Predict Weather")

# -----------------------------------------------------------------------------
# 5. MAIN PAGE & PREDICTION LOGIC
# -----------------------------------------------------------------------------
st.title("🌦️ Intelligent Weather Forecasting System")
st.markdown("### A 2-Stage Machine Learning Pipeline")
st.info("This app uses a Classifier to predict **Rain Probability**, and if positive, a Regressor to estimate **Rainfall Amount**.")

if submitted:
    # 1. Collect Raw Data
    raw_data = {
        'MinTemp': MinTemp, 'MaxTemp': MaxTemp, 'Rainfall': Rainfall, 
        'Evaporation': Evaporation, 'Sunshine': Sunshine,
        'WindGustDir': WindGustDir, 'WindGustSpeed': WindGustSpeed,
        'WindDir9am': WindDir9am, 'WindDir3pm': WindDir3pm,
        'WindSpeed9am': WindSpeed9am, 'WindSpeed3pm': WindSpeed3pm,
        'Humidity9am': Humidity9am, 'Humidity3pm': Humidity3pm,
        'Pressure9am': Pressure9am, 'Pressure3pm': Pressure3pm,
        'Cloud9am': Cloud9am, 'Cloud3pm': Cloud3pm,
        'Temp9am': Temp9am, 'Temp3pm': Temp3pm,
        'RainToday': RainToday
    }

    # 2. Preprocess Data for Classifier
    if clf and scaler_clf:
        X_clf_ready = preprocess_input(raw_data, features_clf, scaler_clf)
        
        if X_clf_ready is not None:
            # 3. Make Prediction
            with st.spinner('Calculating probabilities...'):
                rain_prob = clf.predict_proba(X_clf_ready)[0][1] # Probability of Class 1 (Yes)
                rain_pred = clf.predict(X_clf_ready)[0]          # 0 or 1



            # 4. Display Results
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                
                st.metric("Rain Probability", f"{rain_prob*100:.1f}%")
                
            with col_res2:
                if rain_pred == 1: 
                    st.error("🌧️ **Prediction: IT WILL RAIN TOMORROW!**")
                    st.snow() 
                    
                    if reg and scaler_reg:
                       
                        X_reg_ready = preprocess_input(raw_data, features_clf, scaler_reg) 
                        amount = reg.predict(X_reg_ready)[0]
                        
                        st.warning(f"💧 Estimated Rainfall: **{amount:.2f} mm**")
                        
                        
                        progress_val = min(int((amount / 50) * 100), 100)
                        
                        if amount < 2:
                            intensity_text = "Light Drizzle ☁️"
                        elif amount < 10:
                            intensity_text = "Moderate Rain 🌧️"
                        else:
                            intensity_text = "Heavy Rain ⛈️"
                            
                        st.write(f"**Intensity:** {intensity_text}")
                        st.progress(progress_val)
                        
                else:
                    st.success("☀️ **Prediction: NO Rain Expected.**")
                    st.toast('Enjoy the sunny day!', icon='☀️')


