import numpy as np
import itertools
import vectorbt as vbt
from config import strategy_params, INIT_CASH
from strategies import build_signals

def walk_forward_optimize(price, strat, train_window=756, test_window=126):
    param_grid = list(itertools.product(*strategy_params[strat].values()))
    best_params = None
    best_score = -np.inf

    for params in param_grid:
        param_dict = dict(zip(strategy_params[strat].keys(), params))
        oos_returns = []
        start = 0
        while start + train_window + test_window <= len(price):
            test_slice = price.iloc[start + train_window:start + train_window + test_window]
            entries, exits = build_signals(test_slice, strat, param_dict)
            pf = vbt.Portfolio.from_signals(test_slice, entries, exits, init_cash=INIT_CASH, fees=0.001)
            oos_returns.append(pf.total_return())
            start += test_window
        avg_return = np.mean(oos_returns) if oos_returns else -np.inf
        if avg_return > best_score:
            best_score = avg_return
            best_params = param_dict
    return best_params, best_score

def run_backtest(price, strat, params):
    entries, exits = build_signals(price, strat, params)
    pf = vbt.Portfolio.from_signals(price, entries, exits, init_cash=INIT_CASH, fees=0.001)
    return pf

def stack_strategies(price, strategies_with_params):
    entry_stack = pd.Series(False, index=price.index)
    exit_stack = pd.Series(False, index=price.index)
    for strat, params in strategies_with_params.items():
        entry, exit = build_signals(price, strat, params)
        entry_stack |= entry
        exit_stack |= exit
    pf = vbt.Portfolio.from_signals(price, entry_stack, exit_stack, init_cash=INIT_CASH, fees=0.001)
    return pf