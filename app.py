import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, make_scorer

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(page_title="K-Fold Cross Validation", layout="centered")
st.title("🔁 K-Fold Cross Validation Demo")

# -------------------------------
# Load dataset
# -------------------------------
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=";")
    return df

data = load_data()

# -------------------------------
# Show dataset
# -------------------------------
st.subheader("📊 Dataset Overview")
st.write("Shape of dataset:", data.shape)
st.dataframe(data.head())

# -------------------------------
# Features & target
# -------------------------------
X = data.drop("quality", axis=1)
y = data["quality"]

# -------------------------------
# Model
# -------------------------------
model = LogisticRegression(max_iter=5000)

# -------------------------------
# K-Fold setup
# -------------------------------
st.subheader("⚙️ K-Fold Configuration")
k = st.slider("Select number of folds (K)", min_value=2, max_value=10, value=5)

kf = KFold(n_splits=k, shuffle=True, random_state=42)

# -------------------------------
# Scoring
# -------------------------------
scoring = make_scorer(mean_squared_error, greater_is_better=False)

# -------------------------------
# Run Cross Validation
# -------------------------------
if st.button("Run K-Fold Cross Validation"):
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)

    st.subheader("📈 Cross Validation Results")
    st.write("MSE for each fold:")
    st.write(cv_scores)

    st.success(f"Average MSE: {np.mean(cv_scores):.4f}")
