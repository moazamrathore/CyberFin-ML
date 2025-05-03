import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, silhouette_score
import datetime
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# --- Styling ---
st.set_page_config(page_title="CyberFinance", layout="wide", page_icon="📈")
st.markdown("""
    <style>
        body { background-color: #0a0a0f; color: #00ff9d; }
        .stApp { background-color: #0a0a0f; }
        h1, h2, h3 { color: #00ccff; text-shadow: 0 0 10px #00ccff50; }
    </style>
""", unsafe_allow_html=True)

# --- Data Fetching ---
@st.cache_data(ttl=3600)
def fetch_data(ticker, start, end, interval='1d'):
    try:
        data = yf.download(ticker, start=start, end=end, interval=interval)
        if data.empty:
            st.warning("No data found. Showing demo data.")
            return generate_sample_data()
        return data
    except:
        st.error("Failed to fetch data, showing sample instead.")
        return generate_sample_data()

def generate_sample_data():
    dates = pd.date_range(datetime.datetime.today() - datetime.timedelta(days=100), periods=100)
    close = np.linspace(100, 150, 100) + np.random.normal(0, 2, 100)
    return pd.DataFrame({
        'Open': close * np.random.uniform(0.98, 1.02, 100),
        'High': close * np.random.uniform(1.00, 1.05, 100),
        'Low': close * np.random.uniform(0.95, 1.00, 100),
        'Close': close,
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)

# --- Indicators ---
def add_indicators(df):
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df.dropna(inplace=True)
    return df

def engineer_features(df):
    df['Target_Reg'] = df['Close'].shift(-1)
    df['Target_Cls'] = (df['Target_Reg'] > df['Close']).astype(int)
    for i in range(1, 6):
        df[f'Lag_{i}'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    return df

# --- ML Training ---
def train_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds, mean_squared_error(y_test, preds) ** 0.5

def train_classification(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds, accuracy_score(y_test, preds)

def train_clustering(X):
    scores = []
    for k in range(2, 5):
        km = KMeans(n_clusters=k, random_state=42).fit(X)
        score = silhouette_score(X, km.labels_)
        scores.append((k, score))
    best_k = max(scores, key=lambda x: x[1])[0]
    final_km = KMeans(n_clusters=best_k, random_state=42).fit(X)
    return final_km, best_k

# --- Visualizations ---
def plot_price(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name='Price'))
    if 'MA20' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA20'))
    if 'MA50' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='MA50'))
    fig.update_layout(template='plotly_dark', height=600)
    return fig

# --- App UI ---
st.title("📈 CyberFinance Trading & ML Platform")

col1, col2 = st.columns([2, 1])
with col1:
    ticker = st.text_input("Enter Ticker (e.g. AAPL, BTC-USD)", "AAPL")
with col2:
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

start = st.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=90))
end = st.date_input("End Date", datetime.date.today())

if st.button("Fetch Data"):
    df = fetch_data(ticker, start, end, interval)
    st.subheader(f"Raw Data: {ticker}")
    st.dataframe(df.tail())

    df = add_indicators(df)
    df = engineer_features(df)
    st.plotly_chart(plot_price(df), use_container_width=True)

    st.subheader("Choose ML Model")
    ml_type = st.selectbox("Model", ["None", "Linear Regression", "Logistic Regression", "K-Means Clustering"])

    if ml_type != "None":
        features = ['Close', 'MA20', 'MA50', 'RSI', 'MACD'] + [f'Lag_{i}' for i in range(1, 6)]
        X = df[features]
        if ml_type == "Linear Regression":
            y = df['Target_Reg']
            X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
            _, preds, rmse = train_regression(X_train, X_test, y_train, y_test)
            st.write(f"RMSE: {rmse:.2f}")
            st.line_chart(pd.DataFrame({'Actual': y_test, 'Predicted': preds}, index=y_test.index))
        elif ml_type == "Logistic Regression":
            y = df['Target_Cls']
            X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
            _, preds, acc = train_classification(X_train, X_test, y_train, y_test)
            st.write(f"Accuracy: {acc:.2%}")
            st.bar_chart(pd.DataFrame({'Actual': y_test, 'Predicted': preds}, index=y_test.index))
        else:
            model, best_k = train_clustering(StandardScaler().fit_transform(X))
            st.write(f"Optimal Clusters: {best_k}")
            st.scatter_chart(pd.DataFrame(StandardScaler().fit_transform(X), columns=features))

    # --- Download Section ---
    st.subheader("📥 Download Options")

    # CSV Download
    csv = df.to_csv().encode('utf-8')
    st.download_button(
        label="📄 Download Data as CSV",
        data=csv,
        file_name=f'{ticker}_data.csv',
        mime='text/csv'
    )

    # PDF Report
    def generate_pdf_report(data, filename="report.pdf"):
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        c.setFont("Helvetica", 12)
        c.drawString(30, 750, f"CyberFinance Report for {ticker}")
        c.drawString(30, 735, f"Date Range: {start} to {end}")
        c.drawString(30, 720, f"Interval: {interval}")
        c.drawString(30, 700, "Sample Close Prices:")

        for i, (date, row) in enumerate(data.tail(10).iterrows()):
            y = 680 - (i * 15)
            c.drawString(30, y, f"{date.strftime('%Y-%m-%d')}: ${row['Close']:.2f}")

        c.save()
        buffer.seek(0)
        return buffer

    pdf_bytes = generate_pdf_report(df)
    st.download_button(
        label="🧾 Download Report as PDF",
        data=pdf_bytes,
        file_name=f'{ticker}_report.pdf',
        mime='application/pdf'
    )
