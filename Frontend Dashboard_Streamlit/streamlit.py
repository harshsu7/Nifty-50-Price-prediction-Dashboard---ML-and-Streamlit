import streamlit as st
import time
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Nifty Fifty Dashboard", layout="wide")

MODEL_PATH = "C:\\Users\\ADMIN\\Downloads\\xgb_nifty_model.pkl"  

FEATURE_COLS = ["MA_5", "MA_10", "lag_1", "lag_2", "lag_3", "volatility", "Volume"]

@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        features = obj.get("features", FEATURE_COLS)
    else:
        model = obj
        features = FEATURE_COLS

    return model, features


def load_data(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    required = {"Date", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    df = df.dropna(subset=["Date", "Close", "Volume"]).copy()
    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
    return df


def make_features(df, ma_short=5, ma_long=10, vol_window=5):
    data = df.copy()

    data["MA_5"] = data["Close"].rolling(ma_short).mean().shift(1)
    data["MA_10"] = data["Close"].rolling(ma_long).mean().shift(1)
    data["lag_1"] = data["Close"].shift(1)
    data["lag_2"] = data["Close"].shift(2)
    data["lag_3"] = data["Close"].shift(3)
    data["volatility"] = data["Close"].rolling(vol_window).std().shift(1)

    data = data.dropna().reset_index(drop=True)
    return data


def evaluate_on_uploaded_data(model, featured_df, features):
    X = featured_df[features]
    y = featured_df["Close"]

    preds = model.predict(X)

    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    mae = float(mean_absolute_error(y, preds))
    r2 = float(r2_score(y, preds))

    out = featured_df[["Date", "Close"]].copy()
    out["Predicted"] = preds
    out.rename(columns={"Close": "Actual"}, inplace=True)

    return rmse, mae, r2, out


def predict_next_day(model, featured_df, features):
    latest_row = featured_df.iloc[[-1]][features]
    pred = model.predict(latest_row)[0]
    return float(pred)


st.title("📈 NIFTY Prediction Dashboard")
st.write("Upload your dataset and use your saved `.pkl` model for predictions.")

with st.sidebar:
    st.header("Settings")
    ma_short = st.slider("Short moving average window", 3, 20, 5)
    ma_long = st.slider("Long moving average window", 5, 50, 10)
    vol_window = st.slider("Volatility window", 3, 20, 5)

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded_file is None:
    st.info("Upload a file with columns: Date, Close, Volume")
    st.stop()

try:
    model, model_features = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model from {MODEL_PATH}: {e}")
    st.stop()

try:
    raw_df = load_data(uploaded_file)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

featured_df = make_features(raw_df, ma_short=ma_short, ma_long=ma_long, vol_window=vol_window)

missing_features = [col for col in model_features if col not in featured_df.columns]
if missing_features:
    st.error(f"These model features are missing from the prepared data: {missing_features}")
    st.stop()

if len(featured_df) == 0:
    st.error("Not enough rows after feature engineering. Upload more historical data.")
    st.stop()

rmse, mae, r2, pred_df = evaluate_on_uploaded_data(model, featured_df, model_features)
next_close = predict_next_day(model, featured_df, model_features)

last_close = float(featured_df.iloc[-1]["Close"])
last_date = pd.to_datetime(featured_df.iloc[-1]["Date"])
next_date = last_date + pd.Timedelta(days=1)

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows Used", f"{len(featured_df):,}")
c2.metric("RMSE", f"{rmse:,.2f}")
c3.metric("MAE", f"{mae:,.2f}")
c4.metric("R²", f"{r2:,.4f}")

st.subheader("Next-day Prediction")
p1, p2, p3 = st.columns(3)
p1.metric("Latest Close", f"{last_close:,.2f}")
p2.metric("Predicted Next Close", f"{next_close:,.2f}")
p3.metric("Predicted Change", f"{next_close - last_close:,.2f}")
st.caption(f"Latest date in data: {last_date.date()} | Forecast date: {next_date.date()}")

left, right = st.columns([1.4, 1])

with left:
    st.subheader("Close Price History")
    st.line_chart(raw_df.set_index("Date")[["Close"]], use_container_width=True)

    st.subheader("Actual vs Predicted")
    chart_df = pred_df.set_index("Date")[["Actual", "Predicted"]]
    st.line_chart(chart_df, use_container_width=True)

with right:
    st.subheader("Latest Engineered Features")
    st.dataframe(
        featured_df.tail(1)[["Date"] + model_features + ["Close"]],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Prepared Data Preview")
    st.dataframe(featured_df.tail(20), use_container_width=True, hide_index=True)

# Optional prediction download
csv = pred_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Predictions CSV",
    data=csv,
    file_name="nifty_predictions.csv",
    mime="text/csv",
)

st.code("streamlit run app.py", language="bash")
