import yfinance as yf
import pandas as pd
import requests

# === API KEYS ===
ALPHA_KEY = "2F6D8A5BI2BTG7QV"   # get free at https://www.alphavantage.co
FMP_KEY   = "BCsXgMHJYdsOpjiM2gB4E9NGS9utzlAj"  # get free at https://financialmodelingprep.com/developer


# === Primary fetch: Yahoo Finance ===
def fetch_yahoo(ticker, start=None, end=None, interval="1d"):
    try:
        if start and end:
            df = yf.download(
                ticker, start=start, end=end,
                interval=interval, auto_adjust=True, progress=False
            )
        else:
            df = yf.download(
                ticker, period="max",
                interval=interval, auto_adjust=True, progress=False
            )

        if df.empty or "Close" not in df.columns:
            return pd.Series(dtype=float)
        return df["Close"].dropna()
    except Exception as e:
        print(f"[Yahoo ERROR] {ticker}: {e}")
        return pd.Series(dtype=float)


# === Fallback 1: Alpha Vantage (via requests) ===
def fetch_alpha(ticker, start=None, end=None):
    try:
        url = (
            f"https://www.alphavantage.co/query?"
            f"function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}"
            f"&outputsize=full&apikey={ALPHA_KEY}"
        )
        r = requests.get(url)
        data = r.json().get("Time Series (Daily)", {})
        if not data:
            return pd.Series(dtype=float)

        df = pd.DataFrame(data).T
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        close = df["5. adjusted close"].astype(float)

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


# === Unified function for stocks ===
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


# === Macro symbols mapping ===
MACRO_SYMBOLS = {
    "VIX": "^VIX",          # Volatility Index
    "US10Y": "^TNX",        # 10Y Treasury Yield
    "DXY": "DX-Y.NYB",      # US Dollar Index
    "SPY": "SPY"            # S&P 500 ETF
}


def get_macro_data(start=None, end=None, selection=None):
    """
    Fetch macroeconomic data with fallback logic.
    Tries Yahoo first, then Alpha Vantage, then FMP.
    Returns dict of {name: pd.Series}.
    """
    selection = selection or ["VIX", "US10Y", "DXY", "SPY"]
    macro_data = {}

    for name in selection:
        symbol = MACRO_SYMBOLS.get(name)
        if not symbol:
            continue

        # Try Yahoo
        series = fetch_yahoo(symbol, start, end)
        if not series.empty:
            print(f"[INFO] Yahoo returned {len(series)} rows for {name}")
            macro_data[name] = series
            continue

        # Try Alpha Vantage
        series = fetch_alpha(symbol, start, end)
        if not series.empty:
            print(f"[INFO] Alpha Vantage returned {len(series)} rows for {name}")
            macro_data[name] = series
            continue

        # Try FMP
        series = fetch_fmp(symbol, start, end)
        if not series.empty:
            print(f"[INFO] FMP returned {len(series)} rows for {name}")
            macro_data[name] = series
            continue

        print(f"[FAIL] No data for macro {name}")
        macro_data[name] = pd.Series(dtype=float)

    return macro_data