# 📈 Stock Market Direction Prediction

An AI-powered Stock Market Direction Prediction system that predicts whether a stock price is likely to move **Up** or **Down** using Machine Learning and Deep Learning models. The project combines feature engineering, data preprocessing, model training, interactive visualizations, and a Streamlit web application to provide an end-to-end stock market prediction platform.

---

## 🚀 Features

- 📊 Stock Market Direction Prediction (Up/Down)
- 🧠 Deep Learning model built with PyTorch
- ⚡ CatBoost and LightGBM models for comparison
- 📈 Feature Engineering using technical indicators
- 📉 Interactive charts and visualizations
- 🔍 SHAP Explainability for model predictions
- 🤖 AI-powered market insights using Google Gemini
- 💾 SQLite database for prediction history
- 🌐 User-friendly Streamlit web interface

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Machine Learning & Deep Learning
- PyTorch
- CatBoost
- LightGBM
- XGBoost
- Scikit-learn

### Data Processing
- Pandas
- NumPy

### Visualization
- Matplotlib
- Plotly
- SHAP

### Web Framework
- Streamlit

### Database
- SQLite

### AI Integration
- Google Gemini API

---

## 📂 Project Structure

```
StockMarketDirectionPrediction/
│
├── app/
│   ├── app.py
│   └── database.db
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── best_model.pth
│   ├── lightgbm_model.pkl
│   ├── catboost_model.cbm
│   ├── scalers.pkl
│   ├── stock_map.pkl
│   └── evaluation plots
│
├── notebooks/
│   └── analysis.ipynb
│
├── scripts/
│
├── src/
│
├── main.py
├── requirements.txt
└── .gitignore
```

---

## 📊 Dataset

The project uses historical stock market data containing information such as:

- Open Price
- High Price
- Low Price
- Close Price
- Volume
- VIX
- Daily Returns

### Engineered Features

- SMA (Simple Moving Average)
- Momentum
- RSI (Relative Strength Index)
- Rolling Volatility
- Log Returns
- Volume Change
- Candle Body
- High-Low Range
- Lag Features

---

## ⚙️ Model Pipeline

1. Data Collection
2. Data Preprocessing
3. Feature Engineering
4. Train / Validation Split
5. Deep Learning Model Training
6. CatBoost & LightGBM Training
7. Model Evaluation
8. Prediction
9. Visualization
10. Deployment using Streamlit

---

## 📈 Evaluation Metrics

The project evaluates models using:

- Accuracy
- Mean Squared Error (MSE)
- Filtered Accuracy
- Coverage Score

---

## 💻 Installation

### Clone Repository

```bash
git clone https://github.com/Ramraju04/Stock-Market-Direction-Prediction.git

cd Stock-Market-Direction-Prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Training

```bash
python main.py
```

---

## 🌐 Run Streamlit Application

```bash
streamlit run app/app.py
```

---

## 📊 Visualizations

The project generates:

- Feature Importance
- Confusion Matrix
- Confidence Distribution
- SHAP Explainability Charts

---

## 🤖 AI Assistant

The application integrates **Google Gemini AI** to provide:

- Market insights
- Prediction explanations
- Interactive AI assistance
- Financial analysis support

> Configure your Gemini API key in a `.env` file before using this feature.

Example:

```env
GOOGLE_API_KEY=your_api_key_here
```

---

## 📦 Requirements

Major libraries used:

- Python 3.11+
- PyTorch
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- LightGBM
- XGBoost
- Plotly
- SHAP
- Matplotlib
- Google Generative AI

---

## 📌 Future Improvements

- Real-time stock market data integration
- LSTM and Transformer-based models
- Portfolio optimization
- Buy/Sell recommendation engine
- Risk analysis dashboard
- Cloud deployment
- Multi-stock comparison

---

## 👨‍💻 Author

**Ramraju Bodda**

- 🎓 B.Tech in Artificial Intelligence & Machine Learning
- 💻 AI Engineer | Python Developer | Machine Learning Enthusiast

### GitHub

https://github.com/Ramraju04

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
