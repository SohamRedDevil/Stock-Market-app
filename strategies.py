import pandas as pd
import vectorbt as vbt

def _to_series(x, index):
    """Force any DataFrame/array into a boolean Series aligned to index."""
    if isinstance(x, pd.DataFrame):
        return x.any(axis=1).reindex(index, fill_value=False)
    if isinstance(x, pd.Series):
        return x.reindex(index, fill_value=False)
    return pd.Series(x, index=index)

def build_signals(price, strat, params):
    """
    Build entry/exit signals for a given strategy.
    Always returns (entries, exits) as boolean Series aligned to price.index.
    """
    try:
        if strat == "MA":
            fast = vbt.MA.run(price, window=params['fast']).ma.iloc[:, 0]
            slow = vbt.MA.run(price, window=params['slow']).ma.iloc[:, 0]
            entries = _to_series(fast > slow, price.index)
            exits   = _to_series(fast < slow, price.index)
            return entries, exits

        elif strat == "RSI":
            rsi = vbt.RSI.run(price, window=params['window']).rsi.iloc[:, 0]
            overbought = params.get("overbought", 70)
            oversold   = params.get("oversold", 30)
            entries = _to_series(rsi < oversold, price.index)
            exits   = _to_series(rsi > overbought, price.index)
            return entries, exits

        elif strat == "MACD":
            macd = vbt.MACD.run(
                price,
                fast_window=params['fast_window'],
                slow_window=params['slow_window'],
                signal_window=params['signal_window']
            )
            entries = _to_series(macd.macd > macd.signal, price.index)
            exits   = _to_series(macd.macd < macd.signal, price.index)
            return entries, exits

        elif strat == "Bollinger":
            bb = vbt.BBANDS.run(
                 price,
                 window=params['window'],
                 std=params.get('std', 2)
            entries = (price < bb.lower.iloc[:, 0]).astype(bool)
            exits   = (price > bb.upper.iloc[:, 0]).astype(bool)
            return entries, exits
        

       elif strat == "Breakout":
            roll_max = price.rolling(params['window']).max()
            roll_min = price.rolling(params['window']).min()
            entries = _to_series(price > roll_max.shift(1), price.index)
            exits   = _to_series(price < roll_min.shift(1), price.index)
            return entries, exits

        elif strat == "Momentum":
            mom = price.pct_change(params['window'])
            entries = _to_series(mom > 0, price.index)
            exits   = _to_series(mom < 0, price.index)
            return entries, exits

        elif strat == "MeanReversion":
            mean = price.rolling(params['window']).mean()
            std  = price.rolling(params['window']).std()
            z    = (price - mean) / std
            entries = _to_series(z < -params['zscore'], price.index)
            exits   = _to_series(z > params['zscore'], price.index)
            return entries, exits

        else:
            return pd.Series(False, index=price.index), pd.Series(False, index=price.index)

    except Exception as e:
        print(f"[Strategy ERROR] {strat}: {e}")
        return pd.Series(False, index=price.index), pd.Series(False, index=price.index)