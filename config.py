# Strategy parameter grids
strategy_params = {
    'MA': {'fast': [5, 10, 20], 'slow': [50, 100, 200]},
    'RSI': {'window': [10, 14, 20],'overbought':[70],'oversold':[30]},
    'MACD': {'fast': [12], 'slow': [26], 'signal': [9]},
    'Bollinger': {'window': [20, 30], 'alpha': [1.5, 2]},
    'Breakout': {'window': [20, 50]},
    'Momentum': {'window': [5, 10, 20]},
    'MeanReversion': {'window': [20], 'zscore': [1.5, 2, 2.5]}
}

# Weights for scoring
TECH_WEIGHT = 0.6
FUND_SENT_WEIGHT = 0.4

# Default capital
INIT_CASH = 100_000