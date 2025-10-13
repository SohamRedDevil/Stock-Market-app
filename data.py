import yfinance as yf
import pandas as pd

MACRO_SYMBOLS = {
    "VIX": "^VIX",          # Volatility Index
    "US10Y": "^TNX",        # 10Y Treasury Yield
    "DXY": "DX-Y.NYB",      # US Dollar Index
    "SPY": "SPY"            # S&P 500 ETF
}

def get_price_data(ticker, start=None, end=None, period="10y", interval="1d", retries=2):
    for attempt in range(retries):
        try:
            if start and end:
                df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
            else:
                df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)

            if df.empty or "Close" not in df.columns:
                print(f"[WARN] Attempt {attempt+1}: No data for {ticker}")
                if attempt == 0:
                    start, end = None, None
                    period = "max"
                    continue
                return pd.Series(dtype=float)

            return df["Close"].dropna()
        except Exception as e:
            print(f"[ERROR] Attempt {attempt+1}: Failed to fetch {ticker}: {e}")
            if attempt == retries - 1:
                return pd.Series(dtype=float)

def get_macro_data(start=None, end=None, selection=None):
    selection = selection or ["VIX", "US10Y", "DXY", "SPY"]
    macro_data = {}
    for name in selection:
        symbol = MACRO_SYMBOLS.get(name)
        if not symbol:
            continue
        series = get_price_data(symbol, start=start, end=end)
        macro_data[name] = series
    return macro_data