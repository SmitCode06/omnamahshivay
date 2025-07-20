import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from dataset_loader import load_diabetes_data
from model import get_classifier, train_and_evaluate_model

st.title("🩺 Diabetes Prediction Dashboard")

df = load_diabetes_data()

st.sidebar.header("🔍 Choose Model & Settings")
classifier_name = st.sidebar.selectbox("Select Classifier", ("SVM", "Random Forest", "Naive Bayes"))
test_size = st.sidebar.slider("Test size (%)", 10, 50, 20)

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())
st.write("Shape:", df.shape)
st.write("Summary Stats:", df.describe())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

if y.value_counts().min() < 2:
    st.error("⚠️ Not enough data for stratified split. One of the classes has <2 samples.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=42, stratify=y)

model = get_classifier(classifier_name)
model, acc, y_pred = train_and_evaluate_model(model, X_train, X_test, y_train, y_test)

st.write("📊 Class Distribution:")
st.bar_chart(y.value_counts())

st.subheader("✅ Model Results")
st.write(f"**Accuracy:** {acc:.2f}")

# Confusion Matrix (using matplotlib)
st.subheader("📉 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
cax = ax_cm.matshow(cm, cmap='Blues')
fig_cm.colorbar(cax)
for (i, j), val in np.ndenumerate(cm):
    ax_cm.text(j, i, str(val), ha='center', va='center')
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
ax_cm.set_xticks([0, 1])
ax_cm.set_yticks([0, 1])
ax_cm.set_xticklabels(["No", "Yes"])
ax_cm.set_yticklabels(["No", "Yes"])
st.pyplot(fig_cm)

# Feature Distributions by Outcome (Matplotlib)
st.subheader("📊 Feature Distributions")
feature = st.selectbox("Choose Feature to Visualize", X.columns)
fig_feat, ax_feat = plt.subplots()
for outcome_class in y.unique():
    subset = df[df["Outcome"] == outcome_class]
    ax_feat.hist(subset[feature], bins=20, alpha=0.5, label=f"Outcome: {outcome_class}")
ax_feat.set_title(f"{feature} Distribution by Outcome")
ax_feat.set_xlabel(feature)
ax_feat.set_ylabel("Count")
ax_feat.legend()
st.pyplot(fig_feat)

# Correlation Heatmap (Matplotlib)
st.subheader("📈 Feature Correlation Heatmap")
fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
corr_matrix = df.corr()
cax = ax_corr.matshow(corr_matrix, cmap="coolwarm")
fig_corr.colorbar(cax)
ax_corr.set_xticks(np.arange(len(corr_matrix.columns)))
ax_corr.set_yticks(np.arange(len(corr_matrix.columns)))
ax_corr.set_xticklabels(corr_matrix.columns, rotation=90)
ax_corr.set_yticklabels(corr_matrix.columns)
st.pyplot(fig_corr)
