import pandas as pd
import vectorbt as vbt

def build_signals(price, strat, params):
    try:
        if strat == "MA":
            fast = vbt.MA.run(price, window=params['fast']).ma
            slow = vbt.MA.run(price, window=params['slow']).ma
            return (fast > slow), (fast < slow)
        elif strat == "RSI":
            rsi = vbt.RSI.run(price, window=params['window']).rsi
            return (rsi < 30), (rsi > 70)
        elif strat == "MACD":
            macd = vbt.MACD.run(price, **params)
            return (macd.macd > macd.signal), (macd.macd < macd.signal)
        elif strat == "Bollinger":
            bb = vbt.BBANDS.run(price, **params)
            return (price < bb.lower), (price > bb.upper)
        elif strat == "Breakout":
            roll_max = price.rolling(params['window']).max()
            roll_min = price.rolling(params['window']).min()
            return (price > roll_max.shift(1)), (price < roll_min.shift(1))
        elif strat == "Momentum":
            mom = price.pct_change(params['window'])
            return (mom > 0), (mom < 0)
        elif strat == "MeanReversion":
            mean = price.rolling(params['window']).mean()
            std = price.rolling(params['window']).std()
            z = (price - mean) / std
            return (z < -params['zscore']), (z > params['zscore'])
        else:
            return pd.Series(False, index=price.index), pd.Series(False, index=price.index)
    except Exception:
        return pd.Series(False, index=price.index), pd.Series(False, index=price.index)