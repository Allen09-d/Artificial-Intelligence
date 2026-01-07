import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE  # For handling class imbalance if needed

# Load the dataset
data = pd.read_csv("hospital_readmissions_30k.csv")

# Separate features and target
X = data.drop("readmitted_30_days", axis=1)
y = data["readmitted_30_days"] 

# Convert categorical columns to numbers
X = pd.get_dummies(X, drop_first=True)

X = X.fillna(X.median())

# Check for class imbalance before splitting
print("Original class distribution:")
print(y.value_counts())
print("Class proportions:")
print(y.value_counts(normalize=True))

# Split into training and testing sets (stratify to maintain balance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check class distribution after splitting
print("\nTraining set class distribution:")
print(y_train.value_counts())
print("Training set proportions:")
print(y_train.value_counts(normalize=True))

print("\nTest set class distribution:")
print(y_test.value_counts())
print("Test set proportions:")
print(y_test.value_counts(normalize=True))

minority_proportion = y_train.value_counts(normalize=True).min()
if minority_proportion < 0.3:
    print(f"\nClass imbalance detected (minority proportion: {minority_proportion:.2f}). Applying SMOTE to balance classes.")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("After SMOTE:")
    print(y_train.value_counts())
    print(y_train.value_counts(normalize=True))
else:
    print(f"\nNo significant class imbalance detected (minority proportion: {minority_proportion:.2f}). Proceeding without resampling.")

# Scale the features (important for logistic regression convergence)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the logistic regression model with class weights for additional robustness
model = LogisticRegression(max_iter=1000, class_weight='balanced')  # Balanced weights help with any residual imbalance
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Print evaluation results
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label="Yes"))
print("Recall:", recall_score(y_test, y_pred, pos_label="Yes"))
print("F1 Score:", f1_score(y_test, y_pred, pos_label="Yes"))
print("ROC AUC:", roc_auc_score(y_test.map({"No": 0, "Yes": 1}), y_prob))

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test.map({"No": 0, "Yes": 1}), y_prob)
plt.plot(fpr, tpr, label="Logistic Regression")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Algorithm 1 - Logistic Regression")
plt.legend()
plt.show()