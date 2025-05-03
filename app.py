import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from io import BytesIO
from fpdf import FPDF

# --- Set Page ---
st.set_page_config(page_title="CyberFinance Neo", layout="wide", page_icon="⚡")

# --- Apply Cyberpunk CSS ---
st.markdown("""
<style>
    html, body, [class*="css"] {
        background-color: #0a0a0f !important;
        color: #00ff9d !important;
        font-family: 'Courier New', Courier, monospace;
    }
    h1, h2, h3 {
        color: #00ccff !important;
        text-shadow: 0 0 10px #00ccff90;
    }
    .stButton>button {
        background-color: #6600cc !important;
        color: white;
        border: 1px solid #9933ff;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #9933ff !important;
        border: 1px solid #cc99ff;
    }
</style>
""", unsafe_allow_html=True)

# --- Header with Cyberpunk Branding ---
st.markdown("""
<div style='text-align: center; padding: 10px; background: linear-gradient(90deg, #0a0a0f, #15151e, #0a0a0f); border-radius: 10px;'>
    <h1 style='font-size: 42px;'>⚡ CYBER<span style='color: #ff0066;'>FINANCE</span> NEO</h1>
    <p style='color: #00ff9d; font-size: 16px;'>TradingView-style Market Analysis + ML Predictions</p>
    <img src='https://media.giphy.com/media/RKvvvZ1VQKX6jA0yhc/giphy.gif' width='70%' style='border-radius: 10px; border: 1px solid #9933ff;'/>
</div>
""", unsafe_allow_html=True)
# --- Sidebar Controls ---
st.sidebar.markdown("## 🔧 Controls")
asset_type = st.sidebar.radio("Asset Type", ["Stocks", "Crypto", "Forex"])
ticker = st.sidebar.text_input("Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
interval = st.sidebar.selectbox("Interval", ["1d", "1h", "1wk", "1mo"], index=0)
indicators = st.sidebar.multiselect("Indicators", ["MA20", "MA50", "RSI", "MACD"], default=["MA20", "MACD"])
model_choice = st.sidebar.selectbox("ML Model", ["None", "Linear Regression", "Logistic Regression", "K-Means Clustering"])
load_data = st.sidebar.button("🚀 Load Data")

# --- Fetch Market Data ---
@st.cache_data
def fetch_data(ticker, start, end, interval):
    try:
        df = yf.download(ticker, start=start, end=end, interval=interval)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# --- Download Helpers ---
def convert_df_to_csv(df):
    return df.to_csv(index=True).encode('utf-8')

def generate_pdf_report(metrics: dict, model_name: str) -> BytesIO:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.set_text_color(0, 255, 157)
    pdf.cell(200, 10, txt=f"CyberFinance Report - {model_name}", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(255, 255, 255)
    for key, value in metrics.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True, align="L")
    output = BytesIO()
    pdf.output(output)
    output.seek(0)
    return output
# --- Load Data and Plot ---
if load_data:
    df = fetch_data(ticker, start_date, end_date, interval)
    if df.empty:
        st.warning("No data found for this symbol and date range.")
    else:
        st.success(f"Loaded {len(df)} rows for {ticker}")
        st.dataframe(df.tail())

        # --- Calculate Technical Indicators ---
        if "MA20" in indicators:
            df["MA20"] = df["Close"].rolling(20).mean()
        if "MA50" in indicators:
            df["MA50"] = df["Close"].rolling(50).mean()
        if "RSI" in indicators:
            delta = df["Close"].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = -delta.clip(upper=0).rolling(14).mean()
            rs = gain / loss
            df["RSI"] = 100 - (100 / (1 + rs))
        if "MACD" in indicators:
            df["EMA12"] = df["Close"].ewm(span=12).mean()
            df["EMA26"] = df["Close"].ewm(span=26).mean()
            df["MACD"] = df["EMA12"] - df["EMA26"]
            df["Signal"] = df["MACD"].ewm(span=9).mean()

        # --- Plot Chart ---
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'], name='Price'))

        if "MA20" in df:
            fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20", line=dict(color="cyan")))
        if "MA50" in df:
            fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50", line=dict(color="magenta")))

        fig.update_layout(title=f"{ticker} Price Chart", template="plotly_dark", height=600)
        st.plotly_chart(fig, use_container_width=True)

        # --- Prepare Data for ML ---
        df['Target_Reg'] = df['Close'].shift(-1)
        df['Target_Cls'] = (df['Target_Reg'] > df['Close']).astype(int)
        for i in range(1, 6):
            df[f'Lag_{i}'] = df['Close'].shift(i)
        df.dropna(inplace=True)

        features = ['Close'] + [f'Lag_{i}' for i in range(1, 6)]
        X = df[features]

        # --- Run ML Model ---
        if model_choice == "Linear Regression":
            y = df['Target_Reg']
            X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
            model = LinearRegression()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = mean_squared_error(y_test, preds) ** 0.5
            st.success(f"Linear Regression RMSE: {rmse:.2f}")
            st.line_chart(pd.DataFrame({'Actual': y_test, 'Predicted': preds}, index=y_test.index))

        elif model_choice == "Logistic Regression":
            y = df['Target_Cls']
            X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
            model = LogisticRegression()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            st.success(f"Logistic Regression Accuracy: {acc:.2%}")
            st.bar_chart(pd.DataFrame({'Actual': y_test, 'Predicted': preds}, index=y_test.index))

        elif model_choice == "K-Means Clustering":
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            scores = []
            for k in range(2, 5):
                km = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
                score = silhouette_score(X_scaled, km.labels_)
                scores.append((k, score))
            best_k = max(scores, key=lambda x: x[1])[0]
            final_km = KMeans(n_clusters=best_k, random_state=42).fit(X_scaled)
            st.success(f"Best Clusters: {best_k}")
            df['Cluster'] = final_km.labels_
            st.dataframe(df[['Close', 'Cluster']].tail())
            st.scatter_chart(df[['Lag_1', 'Lag_2', 'Cluster']])
        # --- File Downloads ---
        st.subheader("📥 Download Your Data")
        csv = convert_df_to_csv(df)
        st.download_button("Download CSV", data=csv, file_name=f"{ticker}_data.csv", mime='text/csv')

        if model_choice != "None":
            metrics = {"Model": model_choice}
            if model_choice == "Linear Regression":
                metrics["RMSE"] = round(rmse, 2)
            elif model_choice == "Logistic Regression":
                metrics["Accuracy"] = f"{acc:.2%}"
            elif model_choice == "K-Means Clustering":
                metrics["Best K"] = best_k
            pdf = generate_pdf_report(metrics, model_choice)
            st.download_button("📄 Download Report PDF", data=pdf, file_name="report.pdf", mime="application/pdf")

# --- Cyberpunk Footer ---
st.markdown("""
<br><hr>
<div style='text-align: center; color: #00ff9d; font-size: 13px;'>
    <p>🔮 Powered by <strong>CyberFinance Neo</strong> | Designed with ⚡ by Traders, for Traders</p>
    <p style='color:#9933ff'>Version 1.0 • All Rights Reserved • 2025</p>
    <img src='https://media.giphy.com/media/3o7qDPxorBbvpB1Pby/giphy.gif' width='50%' style='border-radius: 10px; border: 1px solid #9933ff; margin-top: 10px;'/>
</div>
""", unsafe_allow_html=True)
