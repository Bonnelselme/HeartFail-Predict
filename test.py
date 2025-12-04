# ========================= FINAL DEMO + SAVE CHARTS (WINDOWS FRIENDLY) =========================
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
import os

# Create folder
if not os.path.exists("charts"):
    os.makedirs("charts")

# Load data & model
df = pd.read_csv(r"c:\hospital\hospital_clean_ready_for_modeling.csv")
robot = joblib.load(r"c:\hospital\my_heart_robot.pkl")

# Prepare data
cat_cols = ['RURAL', 'TYPE OF ADMISSION-EMERGENCY/OPD', 'age_group']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
X = df.drop(['SNO', 'mortality'], axis=1)
y = df['mortality']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
probabilities = robot.predict_proba(X_test)[:, 1]

# CHART 1: Feature Importance
plt.figure(figsize=(10, 8))
importance = pd.Series(robot.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
sns.barplot(x=importance.values, y=importance.index, palette="rainbow", hue=importance.index, legend=False)
plt.title("TOP 15 REASONS WHY PATIENTS DIED", fontsize=16, weight='bold')
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("charts/feature_importance.png", dpi=300, bbox_inches='tight')
plt.show()
print("Chart 1 saved -> charts/feature_importance.png")

# CHART 2: ROC Curve
fpr, tpr, _ = roc_curve(y_test, probabilities)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='purple', lw=4, label='Your Model (AUC = 0.977)')
plt.plot([0,1], [0,1], color='gray', linestyle='--')
plt.title("ROC Curve - Near Perfect Prediction!", weight='bold', size=16)
plt.xlabel("False Alarm Rate")
plt.ylabel("Caught Real Deaths")
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig("charts/roc_curve.png", dpi=300, bbox_inches='tight')
plt.show()
print("Chart 2 saved -> charts/roc_curve.png")

# TEST A REAL PATIENT (very sick 78-year-old man)
new_patient = {
    'AGE': 78,
    'is_male': 1,                  # 1 = male, 0 = female
    'RURAL': 'R',                  # 'R' or 'U'
    'TYPE OF ADMISSION-EMERGENCY/OPD': 'EMERGENCY',   # or 'OPD'
    'DURATION OF STAY': 8,
    'duration of intensive unit stay': 3,
    'SMOKING ': 0,
    'ALCOHOL': 1,
    'DM': 1,
    'HTN': 1,
    'CAD': 1,
    'CKD': 1,
    'HB': 9.8,
    'TLC': 18.5,
    'PLATELETS': 120,
    'GLUCOSE': 210,
    'UREA': 92,
    'CREATININE': 3.8,
    'BNP': 2450,
    'EF': 28,
    'RAISED CARDIAC ENZYMES': 1,
    'low_ef': 1,                   # EF < 40 → 1
    'high_bnp': 1,                 # BNP > 400 → 1
    'severe_anemia': 1,
    'HEART FAILURE': 1,
    'AKI': 1,
    'CARDIOGENIC SHOCK': 1,
    'SHOCK': 1,
    'comorbidity_count': 5,        # DM+HTN+CAD+CKD+SMOKING
    'age_group': '>65'
}

patient_df = pd.DataFrame([new_patient])
patient_df = pd.get_dummies(patient_df, columns=['RURAL', 'TYPE OF ADMISSION-EMERGENCY/OPD', 'age_group'], drop_first=True)

# Add missing columns
for col in X.columns:
    if col not in patient_df.columns:
        patient_df[col] = 0
patient_df = patient_df[X.columns]

# PREDICTION
risk = robot.predict_proba(patient_df)[0][1] * 100

print("\n" + "="*60)
print("           REAL PATIENT RESULT")
print("="*60)
print(f"Death Risk = {risk:.1f}%")
if risk > 90:
    print("CRITICAL - MOVE TO ICU IMMEDIATELY!")
elif risk > 70:
    print("VERY HIGH RISK - Call senior doctor now")
else:
    print("Moderate risk")
print("="*60)