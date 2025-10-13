import yfinance as yf
import pandas as pd

def get_price_data(ticker, start=None, end=None, period="10y"):
    try:
        df = yf.download(ticker, start=start, end=end, period=period, auto_adjust=True)
        return df["Close"] if "Close" in df.columns else df["Adj Close"]
    except Exception:
        return pd.Series(dtype=float)

def get_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            "PE": info.get("trailingPE"),
            "EPS_GROWTH": info.get("earningsQuarterlyGrowth"),
            "DE_RATIO": info.get("debtToEquity")
        }
    except Exception:
        return {"PE": None, "EPS_GROWTH": None, "DE_RATIO": None}