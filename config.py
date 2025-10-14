strategy_params = {
    "MA": {"fast": [10, 20], "slow": [50, 100]},
    "RSI": {"window": [14], "overbought": [70], "oversold": [30]},
    "MACD": {"fast": [12], "slow": [26], "signal": [9]},
    "Bollinger": {"window": [20], "std": [2]},
    "Breakout": {"window": [20]},
    "Momentum": {"window": [10]},
    "MeanReversion": {"window": [20], "zscore": [1, 2]}
}

# Weights for scoring
TECH_WEIGHT = 0.6
FUND_SENT_WEIGHT = 0.4

# Default capital
INIT_CASH = 100_000