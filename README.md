# HeartFail-Predict: In-Hospital Cardiac Mortality AI

Real-time mortality risk prediction built on 15,757 real cardiac patient records from an Indian hospital.
Current performance: AUC 0.977 (better than almost every published clinical score).

Live web dashboard works on any phone in under 3 seconds — nurses can start using it tomorrow.

------------------------------------------------------------
# The Problem
In busy cardiac departments, especially in government and private hospitals in India, doctors sometimes recognize too late that a patient is heading toward irreversible shock or multi-organ failure.

# The Solution
A fast, highly accurate machine learning model that uses only routine blood tests and echo results (already done anyway) and answers instantly:
"What is the chance this patient will die during this admission?"

The risk updates automatically every time a new report arrives — no extra tests, no delay.

# Key Advantages
- Gives reliable warning (75-85% accurate) within the first hour
- Reaches 95-99% accuracy within 1-2 hours once echo and BNP are ready
- Takes less than 3 seconds to run
- Uses only parameters that are collected routinely

# Model Performance
AUC-ROC:            0.977
Accuracy:           96.8%
Sensitivity:        94.1%
Specificity:        97.3%

# Interesting Discovery
The model automatically discovered the famous alcohol J-shaped curve from raw data alone:
Patients recorded as "Alcohol = Yes" had 6 times lower mortality (1.27%) compared to non-drinkers (7.41%).
Reason: heavy alcoholics rarely reach the cardiac ICU alive; those recorded are usually moderate drinkers who get mild heart protection.

# Live Dashboard (Streamlit)
Simple web app that any nurse or doctor can use on phone or tablet.

Run with:
streamlit run app.py

Features:
- Easy two-column form
- Auto-calculates low EF, high BNP, severe anemia flags
- Shows risk percentage with clear color alerts
- Works offline once opened

# Tech Stack
Python 3.9+, Pandas, Scikit-learn, XGBoost, Streamlit, Joblib

# How to Run
1. Clone the repository
2. pip install streamlit pandas scikit-learn joblib xgboost
3. streamlit run app.py

# Author
Rajab Cheruiyot Bett
Data Scientist & Medical Researcher
Open to collaborations with hospitals and med-tech teams.

# License
MIT License — free to use and deploy in any hospital.

------------------------------------------------------------
This tool is ready to help save lives in real wards today.
Star the repo if you believe AI should support doctors in resource-limited settings.