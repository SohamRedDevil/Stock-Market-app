import numpy as np
import pandas as pd
import itertools
import vectorbt as vbt
from config import strategy_params, INIT_CASH


def _get_build_signals():
    # Local import to avoid circulars/reload issues
    from strategies import build_signals
    return build_signals


def walk_forward_optimize(price, strat, train_window=756, test_window=126):
    """
    Walk-forward optimization: rolling train/test windows to evaluate params.
    """
    build_signals = _get_build_signals()

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
            pf = vbt.Portfolio.from_signals(
                test_slice, entries, exits, init_cash=INIT_CASH, fees=0.001
            )
            oos_returns.append(float(pf.total_return()))
            start += test_window

        avg_return = float(np.mean(oos_returns)) if oos_returns else -np.inf
        if avg_return > best_score:
            best_score = avg_return
            best_params = param_dict

    return best_params, best_score


def run_backtest(price, strat, params):
    """Run a backtest for a single strategy with given params."""
    build_signals = _get_build_signals()
    entries, exits = build_signals(price, strat, params)
    pf = vbt.Portfolio.from_signals(price, entries, exits, init_cash=INIT_CASH, fees=0.001)
    return pf


def stack_strategies(price, strategies_with_params):
    """Combine multiple strategies with OR logic (any entry/exit triggers)."""
    build_signals = _get_build_signals()
    entry_stack = pd.Series(False, index=price.index)
    exit_stack = pd.Series(False, index=price.index)

    for strat, params in strategies_with_params.items():
        entries, exits = build_signals(price, strat, params)
        entry_stack |= entries
        exit_stack |= exits

    pf = vbt.Portfolio.from_signals(price, entry_stack, exit_stack, init_cash=INIT_CASH, fees=0.001)
    return pf


def stack_by_correlation(price, strategies_with_params, lookback=252, corr_threshold=0.3, metric='returns'):
    """
    Greedy stacking: add strategies whose correlation with the current stack
    (returns or signals) is below threshold.
    """
    build_signals = _get_build_signals()
    series_dict = {}
    portfolios = {}

    for strat, params in strategies_with_params.items():
        entries, exits = build_signals(price, strat, params)
        pf = vbt.Portfolio.from_signals(price, entries, exits, init_cash=INIT_CASH, fees=0.001)
        portfolios[strat] = pf

        if metric == 'returns':
            s = pf.daily_returns()
        else:
            e = entries
            if isinstance(e, pd.DataFrame):
                e = e.any(axis=1)
            s = e.astype(int).diff().fillna(0).clip(lower=0)

        s = s[-lookback:]
        series_dict[strat] = s

    chosen = []
    ordered_strats = list(strategies_with_params.keys())
    stack_vector = None

    for strat in ordered_strats:
        s = series_dict[strat]
        if stack_vector is None:
            chosen.append(strat)
            stack_vector = s.copy()
        else:
            corr = float(stack_vector.corr(s)) if len(stack_vector.dropna()) and len(s.dropna()) else 0.0
            if not np.isnan(corr) and abs(corr) < corr_threshold:
                chosen.append(strat)
                stack_vector = pd.concat([stack_vector, s], axis=1).mean(axis=1)

    entry_stack = pd.Series(False, index=price.index)
    exit_stack = pd.Series(False, index=price.index)
    for strat in chosen:
        params = strategies_with_params[strat]
        entries, exits = build_signals(price, strat, params)
        entry_stack |= entries
        exit_stack |= exits

    pf = vbt.Portfolio.from_signals(price, entry_stack, exit_stack, init_cash=INIT_CASH, fees=0.001)
    return pf, chosen