import yfinance as yf
import pandas as pd
import requests
from alpha_vantage.timeseries import TimeSeries

# === API KEYS ===
ALPHA_KEY = "2F6D8A5BI2BTG7QV"   # get free at https://www.alphavantage.co
FMP_KEY   = "BCsXgMHJYdsOpjiM2gB4E9NGS9utzlAj"             # get free at https://financialmodelingprep.com/developer

# === Primary fetch: Yahoo Finance ===
def fetch_yahoo(ticker, start=None, end=None, interval="1d"):
    try:
        if start and end:
            # Only pass start/end
            df = yf.download(ticker, start=start, end=end, interval=interval,
                             auto_adjust=True, progress=False)
        else:
            # Only pass period
            df = yf.download(ticker, period="max", interval=interval,
                             auto_adjust=True, progress=False)

        if df.empty or "Close" not in df.columns:
            return pd.Series(dtype=float)
        return df["Close"].dropna()
    except Exception as e:
        print(f"[Yahoo ERROR] {ticker}: {e}")
        return pd.Series(dtype=float)

# === Fallback 1: Alpha Vantage ===
def fetch_alpha(ticker, start=None, end=None):
    try:
        ts = TimeSeries(key=ALPHA_KEY, output_format="pandas")
        data, _ = ts.get_daily_adjusted(symbol=ticker, outputsize="full")
        close = data["5. adjusted close"].sort_index()
        if start:
            close = close[close.index >= pd.to_datetime(start)]
        if end:
            close = close[close.index <= pd.to_datetime(end)]
        return close
    except Exception as e:
        print(f"[Alpha ERROR] {ticker}: {e}")
        return pd.Series(dtype=float)

# === Fallback 2: Financial Modeling Prep ===
def fetch_fmp(ticker, start=None, end=None):
    try:
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={FMP_KEY}"
        r = requests.get(url)
        data = r.json().get("historical", [])
        if not data:
            return pd.Series(dtype=float)
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        close = df["close"].sort_index()
        if start:
            close = close[close.index >= pd.to_datetime(start)]
        if end:
            close = close[close.index <= pd.to_datetime(end)]
        return close
    except Exception as e:
        print(f"[FMP ERROR] {ticker}: {e}")
        return pd.Series(dtype=float)

# === Unified function ===
def get_price_data(ticker, start=None, end=None):
    # Try Yahoo first
    series = fetch_yahoo(ticker, start, end)
    if not series.empty:
        print(f"[INFO] Yahoo returned {len(series)} rows for {ticker}")
        return series

    # Try Alpha Vantage
    series = fetch_alpha(ticker, start, end)
    if not series.empty:
        print(f"[INFO] Alpha Vantage returned {len(series)} rows for {ticker}")
        return series

    # Try FMP
    series = fetch_fmp(ticker, start, end)
    if not series.empty:
        print(f"[INFO] FMP returned {len(series)} rows for {ticker}")
        return series

    print(f"[FAIL] No data for {ticker}")
    return pd.Series(dtype=float)