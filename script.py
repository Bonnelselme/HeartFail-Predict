import joblib

robot = joblib.load(r"c:\hospital\my_heart_robot.pkl")

print("=== ALL COLUMNS YOUR ROBOT KNOWS (exact spelling) ===\n")
for i, name in enumerate(robot.feature_names_in_, 1):
    print(f"{i:2}. {repr(name)}")   # repr shows hidden spaces perfectly