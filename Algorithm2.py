import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score

# Load the dataset
data = pd.read_csv("hospital_readmissions_30k.csv")

# Separate features and target
X = data.drop("readmitted_30_days", axis=1)
y = data["readmitted_30_days"]  # Labels are 'Yes' and 'No'

# Convert categorical columns to numbers
X = pd.get_dummies(X, drop_first=True)

# Fill missing values with 0
X = X.fillna(0)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features (optional for Random Forest, but kept for consistency)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the random forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Print evaluation results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label="Yes"))
print("Recall:", recall_score(y_test, y_pred, pos_label="Yes"))
print("F1 Score:", f1_score(y_test, y_pred, pos_label="Yes"))
print("ROC AUC:", roc_auc_score(y_test.map({"No": 0, "Yes": 1}), y_prob))

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test.map({"No": 0, "Yes": 1}), y_prob)
plt.plot(fpr, tpr, label="Random Forest")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Algorithm 2 - Random Forest")
plt.legend()
plt.show()