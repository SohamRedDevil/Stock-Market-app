# data.py
import yfinance as yf
import pandas as pd

def get_price_data(ticker, start=None, end=None, period="10y"):
    try:
        if start and end:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        else:
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)

        if df.empty:
            print(f"[WARN] No data returned for {ticker}")
            return pd.Series(dtype=float)

        return df["Close"]
    except Exception as e:
        print(f"[ERROR] Failed to fetch {ticker}: {e}")
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

# Macro proxies via yfinance symbols
MACRO_SYMBOLS = {
    "VIX": "^VIX",          # Volatility Index
    "US10Y": "^TNX",        # 10Y Treasury Yield (index points)
    "DXY": "DX-Y.NYB",      # US Dollar Index (Yahoo symbol)
    "SPY": "SPY"            # SPY ETF as equity benchmark
}

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