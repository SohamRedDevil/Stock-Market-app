# core.py
import numpy as np
import pandas as pd
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
            oos_returns.append(float(pf.total_return()))
            start += test_window

        avg_return = float(np.mean(oos_returns)) if oos_returns else -np.inf
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

def stack_by_correlation(price, strategies_with_params, lookback=252, corr_threshold=0.3, metric='returns'):
    """
    Greedy stacking: start from the highest expected score (walk-forward winner) if known externally,
    otherwise from the first strategy. Add strategies whose metric correlation to the stack is < threshold.

    metric:
      - 'returns': use daily returns from each strategy's portfolio (entries/exits applied)
      - 'signals': use boolean signals (entries) to estimate correlation of firing patterns
    """
    # Build per-strategy series for correlation
    series_dict = {}
    portfolios = {}
    for strat, params in strategies_with_params.items():
        entries, exits = build_signals(price, strat, params)
        pf = vbt.Portfolio.from_signals(price, entries, exits, init_cash=INIT_CASH, fees=0.001)
        portfolios[strat] = pf

        if metric == 'returns':
            s = pf.total_return_series() if hasattr(pf, 'total_return_series') else pf.daily_returns()
        else:
            # entries as int; if DataFrame, reduce row to any=True
            e = entries
            if isinstance(e, pd.DataFrame):
                e = e.any(axis=1)
            s = e.astype(int).diff().fillna(0).clip(lower=0)  # approximate "new signals"
        # align and limit to lookback
        s = s[-lookback:]
        series_dict[strat] = s

    chosen = []
    # If you have external scores, order strategies by expected performance; else keep dict order
    ordered_strats = list(strategies_with_params.keys())

    stack_vector = None
    for strat in ordered_strats:
        s = series_dict[strat]
        if stack_vector is None:
            chosen.append(strat)
            stack_vector = s.copy()
        else:
            # compute correlation with current stack vector
            corr = float(stack_vector.corr(s)) if len(stack_vector.dropna()) and len(s.dropna()) else 0.0
            if not np.isnan(corr) and abs(corr) < corr_threshold:
                chosen.append(strat)
                # update stack vector as average to represent combined behavior
                stack_vector = pd.concat([stack_vector, s], axis=1).mean(axis=1)

    # Combine chosen strategies entries/exits (OR logic)
    entry_stack = pd.Series(False, index=price.index)
    exit_stack = pd.Series(False, index=price.index)
    for strat in chosen:
        params = strategies_with_params[strat]
        entry, exit = build_signals(price, strat, params)
        entry_stack |= entry
        exit_stack |= exit

    pf = vbt.Portfolio.from_signals(price, entry_stack, exit_stack, init_cash=INIT_CASH, fees=0.001)
    return pf, chosen