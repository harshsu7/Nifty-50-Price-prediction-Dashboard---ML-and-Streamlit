import pickle
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="NIFTY Dashboard", layout="wide")


MODEL_PATH = r"C:\Users\ADMIN\Downloads\pkl model\xgb_nifty_model.pkl"   # your saved model
DATA_PATH = r"C:\Users\ADMIN\Downloads\NIFTY50_all.csv"     # <-- put your CSV path here

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


@st.cache_data
def load_data(path):
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

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


def evaluate(model, df, features):
    X = df[features]
    y = df["Close"]

    preds = model.predict(X)

    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    mae = float(mean_absolute_error(y, preds))
    r2 = float(r2_score(y, preds))

    out = df[["Date", "Close"]].copy()
    out["Predicted"] = preds
    out.rename(columns={"Close": "Actual"}, inplace=True)

    return rmse, mae, r2, out


def predict_next(model, df, features):
    latest_row = df.iloc[[-1]][features]
    return float(model.predict(latest_row)[0])

st.title(" NIFTY Prediction Dashboard")

with st.sidebar:
    st.header("Settings")
    ma_short = st.slider("Short MA", 3, 20, 5)
    ma_long = st.slider("Long MA", 5, 50, 10)
    vol_window = st.slider("Volatility Window", 3, 20, 5)

# Load model
try:
    model, model_features = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

# Load data
try:
    raw_df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Data load error: {e}")
    st.stop()

featured_df = make_features(raw_df, ma_short, ma_long, vol_window)

missing = [c for c in model_features if c not in featured_df.columns]
if missing:
    st.error(f"Missing features: {missing}")
    st.stop()

rmse, mae, r2, pred_df = evaluate(model, featured_df, model_features)
next_close = predict_next(model, featured_df, model_features)

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
p3.metric("Change", f"{next_close - last_close:,.2f}")

st.caption(f"Latest date: {last_date.date()} → Prediction for: {next_date.date()}")

# Charts
left, right = st.columns([1.4, 1])

with left:
    st.subheader("Close Price")
    st.line_chart(raw_df.set_index("Date")[["Close"]])

    st.subheader("Actual vs Predicted")
    st.line_chart(pred_df.set_index("Date")[["Actual", "Predicted"]])

with right:
    st.subheader("Latest Features")
    st.dataframe(
        featured_df.tail(1)[["Date"] + model_features + ["Close"]],
        use_container_width=True
    )

    st.subheader("Data Preview")
    st.dataframe(featured_df.tail(20), use_container_width=True)

# Download predictions
csv = pred_df.to_csv(index=False).encode("utf-8")
st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
