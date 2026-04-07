# Step 1: Setup

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Handle imbalance
from imblearn.over_sampling import SMOTE


# Step 2: Load Dataset
# Make sure creditcard.csv is in the same folder as this script

df = pd.read_csv("creditcard.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print(df.head())


# Step 3: Exploratory Data Analysis (EDA)

print("\nClass Distribution:")
print(df['Class'].value_counts())  # 0 = normal, 1 = fraud

# Plot imbalance

sns.countplot(x='Class', data=df)
plt.title("Class Distribution (Imbalanced)")
plt.show()


# Distribution of transaction amounts

plt.figure(figsize=(8,4))

sns.histplot(
    df[df['Class']==0]['Amount'],
    bins=50,
    color='blue',
    label="Normal",
    alpha=0.6
)

sns.histplot(
    df[df['Class']==1]['Amount'],
    bins=50,
    color='red',
    label="Fraud",
    alpha=0.6
)

plt.legend()
plt.title("Transaction Amount Distribution")
plt.show()


# Distribution over time

plt.figure(figsize=(8,4))

sns.histplot(
    df[df['Class']==0]['Time'],
    bins=50,
    color='blue',
    label="Normal",
    alpha=0.6
)

sns.histplot(
    df[df['Class']==1]['Time'],
    bins=50,
    color='red',
    label="Fraud",
    alpha=0.6
)

plt.legend()
plt.title("Transaction Time Distribution")
plt.show()


# Step 4: Train-Test Split + Scaling

X = df.drop("Class", axis=1)
y = df["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Step 4.1: Handle Imbalance with SMOTE

sm = SMOTE(random_state=42)

X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("\nBefore SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_res.value_counts())


# Step 5: Model Training

# Logistic Regression

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train_res, y_train_res)

y_pred_lr = lr.predict(X_test)


# Random Forest

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train_res, y_train_res)

y_pred_rf = rf.predict(X_test)


# XGBoost

xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

xgb.fit(X_train_res, y_train_res)

y_pred_xgb = xgb.predict(X_test)


# Step 6: Evaluation Function

def evaluate_model(y_true, y_pred, model_name):

    print(f"\n=== {model_name} ===")

    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title(f"{model_name} Confusion Matrix")

    plt.show()


evaluate_model(y_test, y_pred_lr, "Logistic Regression")

evaluate_model(y_test, y_pred_rf, "Random Forest")

evaluate_model(y_test, y_pred_xgb, "XGBoost")


# Step 7: ROC-AUC Comparison

plt.figure(figsize=(6,6))

for model, name in [

    (lr, "LR"),

    (rf, "RF"),

    (xgb, "XGB")

]:

    probs = model.predict_proba(X_test)[:,1]

    fpr, tpr, _ = roc_curve(y_test, probs)

    auc = roc_auc_score(y_test, probs)

    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")


plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend()

plt.title("ROC Curves")

plt.show()


# Step 8: Feature Importance

# Random Forest

importances = rf.feature_importances_

indices = np.argsort(importances)[-10:]

plt.figure(figsize=(8,6))

plt.barh(range(len(indices)), importances[indices])

plt.yticks(range(len(indices)), [df.columns[i] for i in indices])

plt.title("Random Forest - Top 10 Important Features")

plt.show()


# XGBoost

xgb_importances = xgb.feature_importances_

indices = np.argsort(xgb_importances)[-10:]

plt.figure(figsize=(8,6))

plt.barh(range(len(indices)), xgb_importances[indices], color="orange")

plt.yticks(range(len(indices)), [df.columns[i] for i in indices])

plt.title("XGBoost - Top 10 Important Features")

plt.show()


# Step 9: Conclusion

print("\nConclusion:")

print("1. Logistic Regression gives a baseline model.")

print("2. Random Forest and XGBoost handle imbalance better after SMOTE.")

print("3. ROC-AUC and feature importance suggest which features are key in detecting fraud.")

print("4. XGBoost usually performs best, but Logistic Regression provides interpretability.")
