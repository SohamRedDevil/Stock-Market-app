import pandas as pd
import vectorbt as vbt

def _to_series(x, index):
    """Force any DataFrame/array into a boolean Series aligned to index."""
    if isinstance(x, pd.DataFrame):
        # Collapse multiple columns with any() across axis=1
        return x.any(axis=1).reindex(index, fill_value=False)
    if isinstance(x, pd.Series):
        return x.reindex(index, fill_value=False)
    return pd.Series(x, index=index)

def build_signals(price, strat, params):
    """
    Return (entries, exits) as boolean Series aligned to price.index.
    Uses vectorbt indicators where applicable and squeezes to Series.
    """
    try:
        if strat == "MA":
            fast = vbt.MA.run(price, window=params['fast']).ma.iloc[:, 0]
            slow = vbt.MA.run(price,window=params['slow']).ma.iloc[:, 0]
            entries = (fast > slow).squeeze()
            exits   = (fast < slow).squeeze()
            return entries, exits

        elif strat == "RSI":
            rsi = vbt.RSI.run(price, window=params['window']).rsi
            overbought = params.get("overbought", 70)
            oversold   = params.get("oversold", 30)
            entries = (rsi < oversold).squeeze()
            exits   = (rsi > overbought).squeeze()
            return entries, exits

        elif strat == "MACD":
            macd = vbt.MACD.run(
            price,
            fast_window=params['fast_window'],
            slow_window=params['slow_window'],
            signal_window=params['signal_window']
            )
            entries = macd.macd > macd.signal
            exits   = macd.macd < macd.signal
            return entries.squeeze(), exits.squeeze()

        elif strat == "Bollinger":
            # Expect params keys: window, std
            bb = vbt.BBANDS.run(price, window=params['window'], std=params.get('std', 2))
            entries = (price < bb.lower).squeeze()
            exits   = (price > bb.upper).squeeze()
            return entries, exits

        elif strat == "Breakout":
            # Expect param: window
            roll_max = price.rolling(params['window']).max()
            roll_min = price.rolling(params['window']).min()
            entries = (price > roll_max.shift(1)).squeeze()
            exits   = (price < roll_min.shift(1)).squeeze()
            return entries, exits

        elif strat == "Momentum":
            # Expect param: window
            mom = price.pct_change(params['window'])
            entries = (mom > 0).squeeze()
            exits   = (mom < 0).squeeze()
            return entries, exits

        elif strat == "MeanReversion":
            # Expect params: window, zscore
            mean = price.rolling(params['window']).mean()
            std  = price.rolling(params['window']).std()
            z    = (price - mean) / std
            entries = (z < -params['zscore']).squeeze()
            exits   = (z > params['zscore']).squeeze()
            return entries, exits

        else:
            # Unknown strategy -> no signals
            return pd.Series(False, index=price.index), pd.Series(False, index=price.index)

    except Exception as e:
        print(f"[Strategy ERROR] {strat}: {e}")
        return pd.Series(False, index=price.index), pd.Series(False, index=price.index)