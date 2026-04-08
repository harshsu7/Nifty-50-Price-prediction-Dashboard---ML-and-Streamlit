A real-time Streamlit web application that uses Machine Learning (XGBoost) to predict the next-day closing price of the NIFTY-50 index based on historical technical indicators.
Live Link of Project - https://nifty-50-price-prediction-dashboard---ml-and-app-hxdpchl3ewf5g.streamlit.app/
Next-Day Forecasting: Predicts the next closing price using the latest market data.
Technical Indicators: Automatically calculates Moving Averages (MA5, MA10), Lags, and Volatility.
Interactive Visualizations: Compare actual vs. predicted prices through dynamic line charts.
Model Performance Metrics: Real-time calculation of RMSE, MAE, and $R^2$ Score.
Data Export: Download the prediction results as a CSV file for further analysis.

├── pkl model/
│   └── xgb_nifty_model.pkl      # Trained XGBoost model
├── Frontend Dashboard_Streamlit/
│   └── streamlit.py             # Dashboard application code
├── NIFTY50_all.zip              # Historical dataset (Compressed CSV)
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
