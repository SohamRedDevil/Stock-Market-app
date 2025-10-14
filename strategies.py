import pandas as pd
import vectorbt as vbt

def build_signals(price, strat, params):
    try:
        if strat == "MA":
            fast = vbt.MA.run(price, window=params['fast']).ma
            slow = vbt.MA.run(price, window=params['slow']).ma
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
            macd = vbt.MACD.run(price, **params)
            entries = (macd.macd > macd.signal).squeeze()
            exits   = (macd.macd < macd.signal).squeeze()
            return entries, exits

        elif strat == "Bollinger":
            bb = vbt.BBANDS.run(price, **params)
            entries = (price < bb.lower).squeeze()
            exits   = (price > bb.upper).squeeze()
            return entries, exits

        elif strat == "Breakout":
            roll_max = price.rolling(params['window']).max()
            roll_min = price.rolling(params['window']).min()
            entries = (price > roll_max.shift(1)).squeeze()
            exits   = (price < roll_min.shift(1)).squeeze()
            return entries, exits

        elif strat == "Momentum":
            mom = price.pct_change(params['window'])
            entries = (mom > 0).squeeze()
            exits   = (mom < 0).squeeze()
            return entries, exits

        elif strat == "MeanReversion":
            mean = price.rolling(params['window']).mean()
            std  = price.rolling(params['window']).std()
            z    = (price - mean) / std
            entries = (z < -params['zscore']).squeeze()
            exits   = (z > params['zscore']).squeeze()
            return entries, exits

        else:
            return pd.Series(False, index=price.index), pd.Series(False, index=price.index)

    except Exception as e:
        print(f"[Strategy ERROR] {strat}: {e}")
        return pd.Series(False, index=price.index), pd.Series(False, index=price.index)