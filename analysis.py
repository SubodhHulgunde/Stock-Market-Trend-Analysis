
import pandas as pd
import numpy as np


def clean_data(data):
    """Clean historical stock data."""

    if data is None or data.empty:
        return pd.DataFrame()

    data = data.copy()

    # Handle columns returned in multiple levels
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.loc[:, ~data.columns.duplicated()]
    data = data.dropna(subset=["Close"])
    data = data.sort_index()
    data = data[~data.index.duplicated(keep="last")]

    return data


def model_data(data):
    """Create useful features for stock trend analysis."""

    data = clean_data(data)

    if data.empty:
        return data

    # Daily percentage return
    data["Daily Return"] = data["Close"].pct_change() * 100

    # Moving averages
    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA50"] = data["Close"].rolling(50).mean()

    # Daily price change
    data["Price Change"] = data["Close"].diff()

    # Annualized rolling volatility
    data["Volatility"] = (
        data["Daily Return"]
        .rolling(20)
        .std() * np.sqrt(252)
    )

    # Cumulative growth of an initial investment of 100
    daily_returns = data["Close"].pct_change().fillna(0)
    data["Growth Index"] = 100 * (1 + daily_returns).cumprod()

    return data


def calculate_metrics(data):
    """Calculate summary statistics."""

    close = data["Close"].dropna()
    returns = data["Daily Return"].dropna()

    if close.empty:
        return {}

    total_return = (close.iloc[-1] / close.iloc[0] - 1) * 100

    volatility = (
        returns.std() * np.sqrt(252)
        if len(returns) > 1 else 0
    )

    running_peak = close.cummax()
    drawdown = (close / running_peak - 1) * 100
    max_drawdown = drawdown.min()

    return {
        "Latest Close": close.iloc[-1],
        "Total Return (%)": total_return,
        "Average Daily Return (%)": returns.mean() if len(returns) else 0,
        "Annualized Volatility (%)": volatility,
        "Maximum Drawdown (%)": max_drawdown,
        "Highest Close": close.max(),
        "Lowest Close": close.min()
    }
