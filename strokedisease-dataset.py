import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Remove ID column if it exists
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# Fill missing BMI with median
if 'bmi' in df.columns:
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    df['bmi'].fillna(df['bmi'].median(), inplace=True)

# Replace missing smoking values with "never smoked"
if 'smoking_status' in df.columns:
    df['smoking_status'].fillna('never smoked', inplace=True)

# First Rule-Based AI (Expert Rule System)
def expert_rule(row):
    if row['hypertension'] == 1 and row['age'] >= 60:
        return 1
    elif row['heart_disease'] == 1 and row['age'] >= 55:
        return 1
    elif row['avg_glucose_level'] >= 160 and row['bmi'] >= 30:
        return 1
    elif row['smoking_status'] in ['formerly smoked', 'smokes'] and row['age'] >= 50:
        return 1
    else:
        return 0

df['Expert_Prediction'] = df.apply(expert_rule, axis=1)

# Second Rule-Based AI (Score-Based Rule System)
def score_rule(row):
    score = 0
    # Age
    if row['age'] >= 65:
        score += 3
    elif row['age'] >= 55:
        score += 2
    elif row['age'] >= 45:
        score += 1
    # Hypertension and heart disease
    if row['hypertension'] == 1:
        score += 3
    if row['heart_disease'] == 1:
        score += 2
    # Glucose and BMI
    if row['avg_glucose_level'] >= 180:
        score += 3
    elif row['avg_glucose_level'] >= 140:
        score += 2
    if row['bmi'] >= 30:
        score += 1
    # Smoking
    if row['smoking_status'] == 'smokes':
        score += 2
    elif row['smoking_status'] == 'formerly smoked':
        score += 1

    return 1 if score >= 4 else 0

df['Score_Prediction'] = df.apply(score_rule, axis=1)

# Evaluation
def evaluate(pred_col):
    total = len(df)
    actual_positive = df['stroke'].sum()
    predicted_positive = df[pred_col].sum()
    correct = (df['stroke'] == df[pred_col]).sum()
    accuracy = correct / total

    true_pos = len(df[(df['stroke'] == 1) & (df[pred_col] == 1)])
    false_pos = len(df[(df['stroke'] == 0) & (df[pred_col] == 1)])
    false_neg = len(df[(df['stroke'] == 1) & (df[pred_col] == 0)])

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1

# Compute metrics for both algorithms
a_acc, a_prec, a_rec, a_f1 = evaluate('Expert_Prediction')
b_acc, b_prec, b_rec, b_f1 = evaluate('Score_Prediction')

# Display results in terminal
print("\nFirst Rule-Based Method — Expert-Rule System")
print(f"Accuracy: {a_acc:.3f}")
print(f"Precision: {a_prec:.3f}")
print(f"Recall: {a_rec:.3f}")
print(f"F1 Score: {a_f1:.3f}")

print("\nSecond Rule-Based Method — Score-Based System")
print(f"Accuracy: {b_acc:.3f}")
print(f"Precision: {b_prec:.3f}")
print(f"Recall: {b_rec:.3f}")
print(f"F1 Score: {b_f1:.3f}")


# Algorithm A Plot
plt.figure(figsize=(6, 4))
plt.bar(['Accuracy', 'Precision', 'Recall', 'F1'], [a_acc, a_prec, a_rec, a_f1], color='skyblue')
plt.title("First Rule-Based Method (Expert-Rule System)")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.tight_layout()
plt.show()

# Algorithm B Plot
plt.figure(figsize=(6, 4))
plt.bar(['Accuracy', 'Precision', 'Recall', 'F1'], [b_acc, b_prec, b_rec, b_f1], color='lightgreen')
plt.title("Second Rule-Based Method (Score-Based System)")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.tight_layout()
plt.show()
