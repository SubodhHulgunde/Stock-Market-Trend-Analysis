
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from datetime import date, timedelta
from analysis import model_data, calculate_metrics


# Page settings
st.set_page_config(
    page_title="Stock Market Trend Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Market Trend Analysis")
st.write(
    "A Data Modeling and Visualization mini project "
    "using historical stock market data."
)

st.caption(
    "Educational analysis only. Historical performance "
    "does not guarantee future results."
)


# Sidebar controls
st.sidebar.header("Analysis Settings")

stocks = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Tata Motors": "TMPV.NS",
    "State Bank of India": "SBIN.NS",
    "Nifty 50 Index": "^NSEI"
}

company = st.sidebar.selectbox(
    "Select a stock",
    list(stocks.keys())
)

ticker = stocks[company]

today = date.today()
default_start = today - timedelta(days=365)

date_range = st.sidebar.date_input(
    "Select date range",
    value=(default_start, today),
    max_value=today
)

if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    st.info("Please select both a start date and an end date.")
    st.stop()

start_date, end_date = date_range

if start_date >= end_date:
    st.error("The start date must be before the end date.")
    st.stop()


# Download historical data
@st.cache_data(ttl=900)
def download_data(symbol, start, end):
    return yf.download(
        symbol,
        start=start,
        end=end + timedelta(days=1),
        auto_adjust=True,
        progress=False
    )


with st.spinner("Downloading historical stock data..."):
    try:
        raw_data = download_data(
            ticker, start_date, end_date
        )
    except Exception as error:
        st.error("Could not download stock data.")
        st.caption(str(error))
        st.stop()


# Prepare data
if raw_data is None or raw_data.empty:
    st.warning(
        "No data found. Check your internet connection, "
        "stock symbol, and selected dates."
    )
    st.stop()

data = model_data(raw_data)

if data.empty:
    st.warning("No valid closing prices are available.")
    st.stop()

metrics = calculate_metrics(data)


# Summary statistics
st.subheader(f"{company} ({ticker})")

st.caption(
    f"Historical data from {data.index.min().date()} "
    f"to {data.index.max().date()}"
)

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Latest Close",
    f"₹{metrics['Latest Close']:,.2f}"
    if ticker != "^NSEI"
    else f"{metrics['Latest Close']:,.2f}"
)

m2.metric(
    "Total Return",
    f"{metrics['Total Return (%)']:.2f}%"
)

m3.metric(
    "Annualized Volatility",
    f"{metrics['Annualized Volatility (%)']:.2f}%"
)

m4.metric(
    "Maximum Drawdown",
    f"{metrics['Maximum Drawdown (%)']:.2f}%"
)


# Chart 1: Closing price and moving averages
st.subheader("1. Stock Price Trend")

fig1, ax1 = plt.subplots(figsize=(12, 5))

ax1.plot(
    data.index, data["Close"],
    label="Closing Price", linewidth=1.8
)

ax1.plot(
    data.index, data["MA20"],
    label="20-Day Moving Average"
)

ax1.plot(
    data.index, data["MA50"],
    label="50-Day Moving Average"
)

ax1.set_xlabel("Date")
ax1.set_ylabel("Price")
ax1.set_title("Closing Price with Moving Averages")
ax1.legend()
ax1.grid(True, alpha=0.3)
fig1.tight_layout()

st.pyplot(fig1)
plt.close(fig1)


# Chart 2: Trading volume
if "Volume" in data.columns:
    st.subheader("2. Trading Volume")

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.bar(data.index, data["Volume"])
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Shares traded")
    ax2.set_title("Daily Trading Volume")
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()

    st.pyplot(fig2)
    plt.close(fig2)


# Chart 3: Daily percentage returns
st.subheader("3. Daily Returns")

fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.plot(data.index, data["Daily Return"], linewidth=0.8)
ax3.axhline(0, linestyle="--")
ax3.set_xlabel("Date")
ax3.set_ylabel("Return (%)")
ax3.set_title("Daily Percentage Return")
ax3.grid(True, alpha=0.3)
fig3.tight_layout()

st.pyplot(fig3)
plt.close(fig3)


# Chart 4: Growth comparison
st.subheader("4. Cumulative Growth Index")

fig4, ax4 = plt.subplots(figsize=(12, 4))
ax4.plot(data.index, data["Growth Index"], linewidth=1.8)
ax4.axhline(100, linestyle="--")
ax4.set_xlabel("Date")
ax4.set_ylabel("Growth Index")
ax4.set_title("Growth of a Hypothetical Initial Value of 100")
ax4.grid(True, alpha=0.3)
fig4.tight_layout()

st.pyplot(fig4)
plt.close(fig4)


# Chart 5: Distribution of daily returns
st.subheader("5. Daily Return Distribution")

returns = data["Daily Return"].dropna()

if not returns.empty:
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    ax5.hist(returns, bins=30, edgecolor="black")
    ax5.set_xlabel("Daily Return (%)")
    ax5.set_ylabel("Number of Trading Days")
    ax5.set_title("Distribution of Daily Returns")
    ax5.grid(True, axis="y", alpha=0.3)
    fig5.tight_layout()

    st.pyplot(fig5)
    plt.close(fig5)
else:
    st.info("Not enough observations to show return distribution.")


# Additional statistics
st.subheader("6. Statistical Summary")

summary = pd.DataFrame(
    {
        "Statistic": list(metrics.keys()),
        "Value": list(metrics.values())
    }
)

st.dataframe(summary, hide_index=True, use_container_width=True)


# Historical data table
st.subheader("7. Historical Stock Data")

st.dataframe(
    data.sort_index(ascending=False),
    use_container_width=True
)


# Download processed data
csv_data = data.to_csv(index_label="Date").encode("utf-8")

st.download_button(
    label="Download Analysis as CSV",
    data=csv_data,
    file_name=f"{ticker.replace('^', '')}_analysis.csv",
    mime="text/csv"
)

st.success("Analysis completed successfully.")
