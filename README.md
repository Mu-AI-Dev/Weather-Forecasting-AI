# 🌦️ Weather Forecasting AI - Rain Prediction System

A machine learning project that predicts **whether it will rain tomorrow** and **how much rainfall to expect** using advanced feature engineering and ensemble models.

---

## 🎯 Project Overview

This project tackles a real-world problem: **Can we predict rain 24 hours in advance?** Using meteorological data from 2,066 observations, I built a dual-prediction system that provides both **classification** (Yes/No) and **regression** (rainfall amount in mm).

### Key Achievements
- ✅ **Dual Model System:** Logistic Regression for classification + Random Forest for rainfall quantity
- ✅ **Advanced Feature Engineering:** Created 10+ derived features (Temperature Change, Humidity Delta, Wind Speed Variations)
- ✅ **Cyclical Encoding:** Transformed wind directions using trigonometric functions (sin/cos) to preserve directionality
- ✅ **Handles Missing Data:** Robust imputation strategy for real-world incomplete datasets

---

## 📊 Dataset

| Feature Category | Examples |
| :--- | :--- |
| **Temperature** | MinTemp, MaxTemp, Temp9am, Temp3pm |
| **Atmospheric** | Pressure9am, Pressure3pm, Humidity9am, Humidity3pm |
| **Wind** | WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm |
| **Weather** | Rainfall, Evaporation, Sunshine, Cloud Cover |
| **Target** | RainTomorrow (Binary), RISK_MM (Continuous) |

**Data Size:** 2,066 samples × 24 original features → **44 engineered features** after preprocessing

---

## 🛠️ Tech Stack

**Languages & Libraries:**
- `Python 3.x`
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - Machine learning models
- `matplotlib`, `seaborn` - Data visualization

**Models Used:**
1. **Logistic Regression** (Classification)
   - Predicts: Will it rain tomorrow? (Yes/No)
   - Class balancing applied for skewed data
   - Max iterations: 1,000

2. **Random Forest Regressor** (Regression)
   - Predicts: How much rain? (mm)
   - Hyperparameters: max_depth=15, min_samples_split=5
   - 100 decision trees ensemble

---

## 🔬 Feature Engineering Highlights

Created intelligent features to capture weather patterns:

- **`TempChange`** = Temp3pm - Temp9am *(Daily temperature swing)*
- **`HumidityChange`** = Humidity3pm - Humidity9am *(Moisture trends)*
- **`PressureChange`** = Pressure3pm - Pressure9am *(Atmospheric stability)*
- **`AvgTemp`**, **`AvgHumidity`**, **`AvgPressure`** *(Daily aggregates)*
- **Wind Direction Encoding:** Converted compass directions (N, E, S, W) into **sin/cos pairs** to preserve circular nature

---

## 📈 Sample Results

| Scenario | Will Rain Tomorrow? | Rain Probability | Predicted Rainfall (mm) |
| :--- | :--- | :--- | :--- |
| Clear & Dry | **No** | 0.33% | 0.00 |
| High Humidity + Pressure Drop | **Yes** | 99.7% | 1.25 |
| Strong Winds + Clouds | **Yes** | 98.0% | 10.21 |

*The model successfully identifies high-risk rain scenarios and estimates rainfall quantity.*

---

## 📁 Project Structure

Weather-Forecasting-AI/
├── data/
│   └── weather_forecasting_dataset.csv     # Raw weather data
├── notebooks/
│   └── Weather-Forecasting.ipynb           # Full analysis & modeling
├── requirements.txt                         # Python dependencies
└── README.md


## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Mu-AI-Dev/Weather-Forecasting-AI.git
   cd Weather-Forecasting-AI

2. **Install dependencies:**
   ```bash
    pip install -r requirements.txt

3. **Open the notebook:**
   ```bash
    jupyter notebook notebooks/Weather-Forecasting.ipynb
    Run all cells to see the full pipeline from data loading to predictions.


## 💡 What I Learned

Feature Engineering Impact: Custom features improved model performance significantly

Model Selection: Why ensemble methods (Random Forest) outperform single models for complex patterns

## 📧 Contact
Muhammad Abdulrahman Ali
🎓 Computer Science Student | Aspiring AI Engineer
📍 Egypt

Email: md.abdelrahmn@gmail.com
LinkedIn: https://www.linkedin.com/in/muhammad-abdelrahaman

