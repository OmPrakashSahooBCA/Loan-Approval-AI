import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("loan_approval_dataset.csv")

# Remove extra spaces from column names and string values
df.columns = df.columns.str.strip()
df['education'] = df['education'].str.strip()
df['self_employed'] = df['self_employed'].str.strip()
df['loan_status'] = df['loan_status'].str.strip()

# Label Encoding
le_education = LabelEncoder()
le_self_employed = LabelEncoder()
le_loan_status = LabelEncoder()

df['education'] = le_education.fit_transform(df['education'])
df['self_employed'] = le_self_employed.fit_transform(df['self_employed'])
df['loan_status'] = le_loan_status.fit_transform(df['loan_status'])

# Print encoding mappings for reference
print("Education encoding:", dict(zip(le_education.classes_, le_education.transform(le_education.classes_))))
print("Self-employed encoding:", dict(zip(le_self_employed.classes_, le_self_employed.transform(le_self_employed.classes_))))
print("Loan status encoding:", dict(zip(le_loan_status.classes_, le_loan_status.transform(le_loan_status.classes_))))

# Features and Target
X = df.drop(['loan_id', 'loan_status'], axis=1)
y = df['loan_status']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = GradientBoostingClassifier()

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Save model and encoders
joblib.dump(model, "model.pkl")
joblib.dump({
    'education': le_education,
    'self_employed': le_self_employed,
    'loan_status': le_loan_status
}, "label_encoders.pkl")

print("Model and encoders saved successfully.")
