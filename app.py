# app.py → run with: streamlit run "c:\hospital\app.py"
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Heart Mortality AI", layout="centered")
st.title("In-Hospital Mortality Prediction")
st.write("Built on 15,757 real cardiac patients | AUC 0.977")

robot = joblib.load(r"c:\hospital\my_heart_robot.pkl")

with st.form("patient_form"):
    st.subheader("Enter Patient Data")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 1, 120, 65)
        male = st.selectbox("Gender", ["Female", "Male"])
        rural = st.selectbox("Area", ["Rural (R)", "Urban (U)"])
        admission = st.selectbox("Admission Type", ["Emergency", "OPD"])
        duration_stay = st.number_input("Days in hospital", 0, 100, 7)
        icu_days = st.number_input("Days in ICU", 0, 100, 0)
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        alcohol = st.selectbox("Alcohol", ["No", "Yes"])
        dm = st.selectbox("Diabetes", ["No", "Yes"])
        htn = st.selectbox("Hypertension", ["No", "Yes"])
        cad = st.selectbox("Coronary Artery Disease", ["No", "Yes"])
        ckd = st.selectbox("Chronic Kidney Disease", ["No", "Yes"])
        
    with col2:
        hb = st.number_input("HB (g/dL)", 0.0, 25.0, 12.0)
        tlc = st.number_input("TLC (x10³)", 0.0, 100.0, 10.0)
        platelets = st.number_input("Platelets", 0, 1000, 250)
        glucose = st.number_input("Glucose", 0, 1000, 140)
        urea = st.number_input("Urea", 0.0, 500.0, 40.0)
        creatinine = st.number_input("Creatinine", 0.0, 20.0, 1.0)
        bnp = st.number_input("BNP", 0, 50000, 470)
        ef = st.number_input("EF (%)", 0, 100, 50)
        cardiac_enzymes = st.selectbox("Raised Cardiac Enzymes", ["No", "Yes"])
        heart_failure = st.selectbox("Heart Failure", ["No", "Yes"])
        aki = st.selectbox("AKI", ["No", "Yes"])
        shock = st.selectbox("Any Shock", ["No", "Yes"])
        cardiogenic_shock = st.selectbox("Cardiogenic Shock", ["No", "Yes"])

    submitted = st.form_submit_button("Predict Mortality Risk")

    if submitted:
        data = {
            'AGE': age, 'is_male': 1 if male=="Male" else 0,
            'DURATION OF STAY': duration_stay, 'duration of intensive unit stay': icu_days,
            'SMOKING ': 1 if smoking=="Yes" else 0, 'ALCOHOL': 1 if alcohol=="Yes" else 0,
            'DM': 1 if dm=="Yes" else 0, 'HTN': 1 if htn=="Yes" else 0,
            'CAD': 1 if cad=="Yes" else 0, 'CKD': 1 if ckd=="Yes" else 0,
            'HB': hb, 'TLC': tlc, 'PLATELETS': platelets, 'GLUCOSE': glucose,
            'UREA': urea, 'CREATININE': creatinine, 'BNP': bnp, 'EF': ef,
            'RAISED CARDIAC ENZYMES': 1 if cardiac_enzymes=="Yes" else 0,
            'low_ef': 1 if ef < 40 else 0, 'high_bnp': 1 if bnp > 400 else 0,
            'severe_anemia': 1 if hb < 8 else 0,
            'HEART FAILURE': 1 if heart_failure=="Yes" else 0,
            'AKI': 1 if aki=="Yes" else 0,
            'CARDIOGENIC SHOCK': 1 if cardiogenic_shock=="Yes" else 0,
            'SHOCK': 1 if shock=="Yes" else 0,
            'comorbidity_count': sum([dm=="Yes", htn=="Yes", cad=="Yes", ckd=="Yes", smoking=="Yes", alcohol=="Yes"]),
            'RURAL_U': 1 if rural=="Urban (U)" else 0,
            'TYPE OF ADMISSION-EMERGENCY/OPD_O': 1 if admission=="OPD" else 0,
            'age_group_<40': 1 if age < 40 else 0,
            'age_group_>65': 1 if age > 65 else 0
        }
        
        df = pd.DataFrame([data])
        for col in robot.feature_names_in_:
            if col not in df.columns:
                df[col] = 0
        df = df[robot.feature_names_in_]
        
        risk = robot.predict_proba(df)[0][1] * 100
        
        st.markdown(f"## Risk of In-Hospital Death: **{risk:.1f}%**")
        if risk > 80:
            st.error("CRITICAL — ICU IMMEDIATELY")
        elif risk > 50:
            st.warning("VERY HIGH RISK — Alert senior doctor")
        elif risk > 20:
            st.info("Moderate risk")
        else:
            st.success("Low risk — Good prognosis")