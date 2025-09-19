
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="Business Sales Predictor", layout="wide")
st.title("📈 Business Sales Predictor")
st.markdown("Upload your historical sales CSV and get short-term forecasts. Columns expected: `date` and `sales`.")

uploaded_file = st.file_uploader("Upload CSV file (date,sales)", type=["csv"])
example_data = st.checkbox("Show example data / download sample CSV")

if example_data:
    st.download_button("Download sample CSV", data=open("sample_data.csv","rb"), file_name="sample_data.csv")
    st.write(pd.read_csv("sample_data.csv").head())

def preprocess(df):
    # Expecting 'date' and 'sales'
    df = df.copy()
    if not ("date" in df.columns and "sales" in df.columns):
        raise ValueError("CSV must contain 'date' and 'sales' columns")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    # Create a numeric time index
    df['t'] = (df['date'] - df['date'].min()).dt.days
    # Lag features
    df['lag_1'] = df['sales'].shift(1)
    df['lag_7'] = df['sales'].shift(7)
    df['rolling_7'] = df['sales'].rolling(window=7, min_periods=1).mean()
    df = df.dropna().reset_index(drop=True)
    return df

def train_predict(df, model_name, horizon):
    df2 = preprocess(df)
    X = df2[['t','lag_1','lag_7','rolling_7']]
    y = df2['sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    if model_name == 'RandomForest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = LinearRegression()
    model.fit(X_train, y_train)
    preds_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds_test)
    # Forecast future
    last_date = df2['date'].max()
    last_sales = df['sales'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, horizon+1)]
    future_t = [(d - df2['date'].min()).days for d in future_dates]
    # Build features for future (use previous predictions for lag)
    future_df = []
    sales_series = list(df['sales'])
    for ft, fd in zip(future_t, future_dates):
        lag_1 = sales_series[-1]
        lag_7 = sales_series[-7] if len(sales_series) >= 7 else sales_series[0]
        rolling_7 = sum(sales_series[-7:]) / min(7, len(sales_series))
        x = [ft, lag_1, lag_7, rolling_7]
        pred = model.predict([x])[0]
        sales_series.append(pred)  # append prediction to be used as lag for next day
        future_df.append({"date": fd, "prediction": float(pred)})
    future_df = pd.DataFrame(future_df)
    # Combine for plotting
    plot_df = pd.concat([
        df[['date','sales']].rename(columns={'sales':'value'}),
        future_df.rename(columns={'prediction':'value'})
    ], ignore_index=True)
    return model, mae, future_df, plot_df

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:", df.head())
        horizon = st.number_input("Forecast horizon (days)", min_value=1, max_value=365, value=30)
        model_name = st.selectbox("Model", ["RandomForest","LinearRegression"])
        if st.button("Train & Predict"):
            with st.spinner("Training model and generating forecast..."):
                model, mae, future_df, plot_df = train_predict(df, model_name, int(horizon))
            st.success(f"Model trained. MAE on test split: {mae:.2f}")
            st.subheader("Forecast")
            st.write(future_df)
            # Plot
            fig, ax = plt.subplots(figsize=(10,4))
            # plot historical
            hist = df[['date','sales']].rename(columns={'sales':'value'})
            ax.plot(hist['date'], hist['value'], label='Actual')
            ax.plot(plot_df['date'].iloc[-len(future_df):], plot_df['value'].iloc[-len(future_df):], marker='o', linestyle='--', label='Forecast')
            ax.set_xlabel("Date")
            ax.set_ylabel("Sales")
            ax.legend()
            st.pyplot(fig)
            csv = future_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download forecast CSV", data=csv, file_name="forecast.csv")
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Upload a CSV to get started. The sample CSV shows daily sales for a small store.")
