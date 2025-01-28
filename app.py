import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

@st.cache_data(persist=True)
def load_data():
    df = pd.read_csv("steam_games.csv")  

    # Remove columns
    drop_columns = ['steam_appid', 'name', 'review_score_desc', 'is_released', 'additional_content', 'release_date']
    df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)

    return df

@st.cache_data(persist=True)
def split(df):
    y = df['positive_percentual']  # Target
    drop_columns = ['positive_percentual']  
    X = df.drop(columns=drop_columns)

    categorical_features = ['developers', 'publishers', 'categories', 'genres', 'platforms']
    for col in categorical_features:
        top_10 = df[col].value_counts().nlargest(10).index
        X[col] = X[col].apply(lambda x: x if x in top_10 else "Other")

    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    numerical_features = ['required_age', 'n_achievements', 'total_reviews', 'price_initial (USD)', 'metacritic']
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    return train_test_split(X, y, test_size=0.3, random_state=42)

@st.cache_data(persist=True)
def train_model(model_name, X_train, y_train):
    if model_name == "Linear Regression":
        model = LinearRegression(n_jobs=-1)
    elif model_name == "Random Forest Regressor":
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def plot_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    st.subheader("📊 Model Performance")
    st.write(f"**MSE:** {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"**R² Score:** {r2_score(y_test, y_pred):.3f}")
    
    fig, ax = plt.subplots()
    ax.hist(y_test - y_pred, bins=30, edgecolor='black', alpha=0.7)
    ax.set_title("Residual Distribution")
    st.pyplot(fig)

df = load_data()
print(df.isnull().sum())  
X_train, X_test, y_train, y_test = split(df)

if st.sidebar.checkbox("Show raw data", False):
    st.subheader("Steam Dataset 2025")
    st.write(df)

st.sidebar.subheader("Choose Model")
model_name = st.sidebar.selectbox("Select Model", ("Linear Regression", "Random Forest Regressor"))

if st.sidebar.button("Train Model"):
    st.subheader(f"Training {model_name}...")
    model = train_model(model_name, X_train, y_train)
    st.success(f"{model_name} trained successfully!")
    plot_metrics(model, X_test, y_test)
