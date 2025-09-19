[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Deploy on Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)


# 📈 Business Sales Predictor

A simple **Streamlit app** that predicts future sales based on historical data.  
Upload your sales CSV (`date` and `sales` columns) → train a model → get forecasts + download results.

---

## 🚀 Features
- Upload sales data (CSV with `date`, `sales`).
- Train two models: **RandomForest** or **LinearRegression**.
- Forecast future sales for up to 365 days.
- Interactive chart of actual vs. predicted sales.
- Download predictions as CSV.

---

## 📂 Project Structure
```
business_sales_predictor/
│── app.py              # Streamlit app
│── requirements.txt    # Python dependencies
│── sample_data.csv     # Example dataset
│── README.md           # Documentation
```

---

## ▶️ Running Locally
1. Clone the repo:
   ```bash
   git clone https://github.com/YOUR-USERNAME/business_sales_predictor.git
   cd business_sales_predictor
   ```

2. Create a virtual environment and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## 🌐 Deploying on Hugging Face Spaces

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) → click **New Space**.  
2. Name it, choose **Streamlit** as SDK, set visibility (Public/Private).  
3. Either:  
   - **Option A:** Upload the ZIP files directly.  
   - **Option B:** Connect your GitHub repo.  
4. Hugging Face will auto-install dependencies from `requirements.txt`.  
5. After setup, your app will be live at:
   ```
   https://huggingface.co/spaces/YOUR-USERNAME/business-sales-predictor
   ```

---

## 📊 Sample Data
We include a `sample_data.csv` file to test quickly:

```csv
date,sales
2024-01-01,200
2024-01-02,220
2024-01-03,210
2024-01-04,230
...
```

---

## 🖼️ Screenshots (add after deployment)
- ![Upload CSV Screenshot](screenshots/upload.png)
- ![Prediction Results Screenshot](screenshots/results.png)

---

## 🔮 Next Steps
- Add **Prophet/ARIMA** for advanced forecasting.  
- Customize with logos, colors, and business branding.  
- Integrate with Google Sheets or POS systems.  

---

👩‍💻 Built with **Python, Streamlit, scikit-learn, and Hugging Face Spaces**.  
