import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score,roc_curve
import matplotlib.pyplot as plt
import  seaborn as sns
import joblib

# Load the data
df = pd.read_csv("hospital_clean_ready_for_modeling.csv")

# Convert the 3 categorical columns into numbers
cat_cols = ['RURAL', 'TYPE OF ADMISSION-EMERGENCY/OPD', 'age_group']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

print("After converting categories new shape:", df.shape)
print("New columns added")
print([col for col in df.columns if 'RURAL' in col or 'ADMISSION' in col or 'age_group' in col])

# Separate features (X) and target(y)
X = df.drop(['SNO', 'mortality'], axis=1)
y = df['mortality']

print("\nFeatures (X) shape :", X.shape)
print("Target (y) shape :", y.shape)
print("Target death rate :", y.mean().round(4))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain size :", X_train.shape)
print("Test size :", X_test.shape)
print("Death rate in test set :", y_test.mean().round(4))

# Create the robot 
robot = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)

# Teach the robot using the training data
robot.fit(X_train, y_train)

# Let the robot guess on the hidden test patients
predictions = robot.predict(X_test)
probabilities = robot.predict_proba(X_test)[:, 1]

# Show how smart it is 

print("HOW GOOD IS OUR ROBOT?")
print("AUC Score =", round(roc_auc_score(y_test, probabilities), 4))
print("\nFull report:")
print(classification_report(y_test, predictions))

# 1. Feature Importance – What matters most?
plt.figure(figsize=(10, 8))
importance = pd.Series(robot.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
sns.barplot(x=importance.values, y=importance.index, palette="rainbow")
plt.title("TOP 15 REASONS WHY PATIENTS DIED (Your Model Discovered This!)", fontsize=14, weight='bold')
plt.xlabel("How Important (bigger = more deadly)")
plt.tight_layout()
plt.show()

# 2. ROC Curve – The “perfect model” line
fpr, tpr, _ = roc_curve(y_test, probabilities)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='purple', lw=3, label=f'Your Model (AUC = 0.977!)')
plt.plot([0,1], [0,1], color='gray', linestyle='--')
plt.title("ROC Curve – How Close to Perfect Is Your Robot?", weight='bold', size=14)
plt.xlabel("False Alarm Rate")
plt.ylabel("Caught Real Deaths Rate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
joblib.dump(robot, r"c:\hospital\my_heart_robot.pkl")
print("Your robot is now saved forever!")

print(df['ALCOHOL'].mean())      # % of patients who drink
print(df[df['ALCOHOL']==1]['mortality'].mean())  # death rate of drinkers
print(df[df['ALCOHOL']==0]['mortality'].mean())  # death rate of non-drinkers