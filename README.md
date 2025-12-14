# 🌦️ Intelligent Weather Forecasting AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://weather-forecasting-ai.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Deployed-success)

> **An End-to-End Machine Learning System** that predicts rain probability and intensity 24 hours in advance, deployed as an interactive web application.

---

## 🔴 Live Demo
### 👉 [Click Here to Try the App](https://weather-forecasting-ai.streamlit.app/)

*(Note: The app is hosted on Streamlit Cloud. If it's asleep, just click "Wake up" and wait a moment!)*

---

## 📸 Project Preview
<!-- Upload a screenshot of your app to your repo and link it here for better engagement -->
![App Screenshot](Capture.jpg)

---

## 🎯 Project Overview

This project tackles a critical meteorological challenge: **Predicting local rainfall with high precision.** unlike standard forecasts, this system uses a **2-Stage Stacked Pipeline** to answer two questions:
1.  **Classification:** *Will it rain tomorrow?* (Yes/No)
2.  **Regression:** *If yes, how much rain will fall?* (mm)

The model was trained on **2,000+ observations** containing complex atmospheric data, utilizing advanced feature engineering to capture non-linear weather patterns.

---

## 🏗️ Architecture & Pipeline

The system is built on a dual-model architecture:

graph LR
A[Input Data] --> B{Classifier Model}
B -- No Rain --> C[Result: Dry Day ☀️]
B -- Rain Expected --> D[Regressor Model]
D --> E[Result: Rain Intensity (mm) 🌧️]


### 1. Classification Stage (Logistic Regression)
- **Goal:** Filter out dry days to reduce noise.
- **Technique:** Balanced class weights to handle data imbalance.

### 2. Regression Stage (Random Forest)
- **Goal:** Estimate rainfall quantity for positive cases.
- **Configuration:** 100 Trees, Max Depth 15, optimized for minimal RMSE.

---

## 📊 Feature Engineering (The Secret Sauce)

Raw data wasn't enough. I engineered **12+ new features** to improve model sensitivity:

| Feature | Logic & Impact |
| :--- | :--- |
| **Cyclical Wind Encoding** | Transformed compass directions (N, NW, S) into `sin` & `cos` pairs to preserve directionality math. |
| **Atmospheric Deltas** | Calculated `PressureChange` (3pm - 9am) to detect incoming fronts. |
| **Daily Swings** | `TempRange` and `HumidityChange` to measure daily volatility. |
| **Imputation Strategy** | Used statistical medians to handle missing sensor data robustly. |

---

## 📁 Project Structure

Weather-Forecasting-AI/
├── app.py # Main Streamlit Application (Frontend & Logic)
├── requirements.txt # Dependencies for deployment
├── models/ # Serialized ML Models & Scalers
│ ├── model_classifier.pkl
│ ├── model_regressor.pkl
│ └── scaler_classifier.pkl
├── notebooks/ # Research & Training
│ └── Weather-Forecasting.ipynb
└── data/ # Raw Dataset


---

## 🚀 How to Run Locally

If you want to run this app on your own machine:

1.  **Clone the repository:**
    ```
    git clone https://github.com/Mu-AI-Dev/Weather-Forecasting-AI.git
    cd Weather-Forecasting-AI
    ```

2.  **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

3.  **Launch the App:**
    ```
    streamlit run app.py
    ```

---

## 📈 Model Performance

| Metric | Score | Notes |
| :--- | :--- | :--- |
| **Accuracy (Test)** | **~85%** | Robust against unseen data |
| **Precision** | **High** | Minimized False Alarms (predicting rain when dry) |
| **Recall** | **Balanced** | Successfully captures most rain events |

---

## 🛠️ Tech Stack

*   **Core:** Python 3.x
*   **ML Libraries:** Scikit-Learn, Pandas, NumPy
*   **Web Framework:** Streamlit (for UI/UX)
*   **Serialization:** Joblib
*   **Visualization:** Matplotlib, Seaborn

---

## 👤 Author

**Muhammad Abdulrahman Ali**
*Computer Science Student | Aspiring AI Engineer*

*   📍 **Location:** Egypt
*   📧 **Email:** [md.abdelrahmn@gmail.com](mailto:md.abdelrahmn@gmail.com)
*   🔗 **LinkedIn:** [muhammad-abdelrahama](https://www.linkedin.com/in/muhammad-abdelrahama)
*   🐙 **GitHub:** [Mu-AI-Dev](https://github.com/Mu-AI-Dev)

---
*If you find this project useful, please give it a ⭐ star!*
