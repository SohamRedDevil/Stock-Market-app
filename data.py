import yfinance as yf
import pandas as pd

def get_price_data(ticker, start=None, end=None, period="10y", interval="1d", retries=2):
    """
    Fetches price data with fallback logic:
    - First tries start/end range
    - If empty, retries with period='max'
    - Returns Close price series or empty Series
    """
    for attempt in range(retries):
        try:
            if start and end:
                df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
            else:
                df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)

            if df.empty or "Close" not in df.columns:
                print(f"[WARN] Attempt {attempt+1}: No data for {ticker}")
                if attempt == 0:
                    # Retry with fallback
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
    """
    selection: list of macro keys from MACRO_SYMBOLS, e.g. ["VIX","US10Y","DXY","SPY"]
    Returns dict: {name: series}
    """
    sel = selection or ["VIX", "US10Y", "DXY", "SPY"]
    out = {}
    for name in sel:
        symbol = MACRO_SYMBOLS.get(name)
        if not symbol:
            continue
        s = get_price_data(symbol, start=start, end=end)
        out[name] = s
    return out