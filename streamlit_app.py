import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from pathlib import Path

from core import walk_forward_optimize, run_backtest, stack_strategies, stack_by_correlation
from config import strategy_params
from data import get_price_data, get_macro_data
from sentiment import get_reddit_sentiment, get_news_sentiment

# --- Setup ---
st.set_page_config(page_title="Multi-Ticker Strategy Lab", layout="wide")
st.title("📊 Multi-Ticker Strategy Lab")

HISTORY_FILE = Path("history.json")

def load_history():
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def recommend_strategy(ticker, history):
    strat_scores = history.get(ticker, {})
    if not strat_scores:
        return None
    return max(strat_scores.items(), key=lambda x: x[1])[0]

def plot_comparison(pf_dict):
    fig = go.Figure()
    for ticker, pf in pf_dict.items():
        cum_returns = pf.cumulative_returns()
        fig.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns.values,
                                 mode='lines', name=ticker))
    fig.update_layout(title="📊 Cumulative Return Comparison", height=500, legend=dict(orientation="h"))
    return fig

def add_macro_overlays(fig, macro_dict, price_index, secondary_y=True):
    # Overlay macro series on secondary y-axis for visibility
    for name, series in macro_dict.items():
        if series is None or series.empty:
            continue
        series = series.reindex(price_index).fillna(method='ffill')
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode='lines',
            name=name,
            yaxis="y2" if secondary_y else "y"
        ))

    fig.update_layout(
        yaxis=dict(title="Price", showgrid=True),
        yaxis2=dict(
            title="Macro",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

# --- Sidebar Inputs ---
tickers = st.text_input("Enter tickers (comma-separated)", value="AAPL,MSFT,NVDA").upper().split(",")
start_date = st.date_input("Start date", value=pd.to_datetime("2018-01-01"))
end_date = st.date_input("End date", value=pd.to_datetime("2023-12-31"))
selected_strategies = st.multiselect("Select strategies", list(strategy_params.keys()), default=["MA", "RSI", "MACD"])

stack_mode = st.selectbox("Stacking mode", ["None", "OR stack", "Correlation-based stack"], index=2)
corr_threshold = st.slider("Correlation threshold (lower = stricter)", min_value=0.0, max_value=0.9, value=0.3, step=0.05)
corr_metric = st.selectbox("Correlation metric", ["returns", "signals"], index=0)

show_sentiment = st.checkbox("Overlay sentiment scores", value=True)
api_key = st.text_input("NewsAPI Key", type="password")

macro_selection = st.multiselect("Macro overlays", ["VIX", "US10Y", "DXY", "SPY"], default=["VIX", "US10Y", "SPY"])

# --- Parameter Tuning ---
st.sidebar.header("🔧 Parameter Tuning (manual override)")
custom_params = {}
for strat in selected_strategies:
    st.sidebar.subheader(strat)
    custom_params[strat] = {}
    for param, values in strategy_params[strat].items():
        custom_params[strat][param] = st.sidebar.selectbox(f"{strat} - {param}", values)

# --- Run Optimization ---
if st.button("Run Strategy Analysis"):
    history = load_history()
    pf_dict = {}
    comparison_rows = []

    for raw_ticker in tickers:
        ticker = raw_ticker.strip()
        if not ticker:
            continue

        st.subheader(f"📈 {ticker}")
        price = get_price_data(ticker, start=start_date, end=end_date)

        # 🔍 Debug: Show price data
        st.write(f"Fetched {len(price)} rows for {ticker}")
        if not price.empty:
            st.line_chart(price)
        else:
            st.warning(f"No price data available for {ticker}")
            continue

        # Walk-forward optimization for each selected strategy
        best_strats = {}
        strat_scores = {}

        for strat in selected_strategies:
            best_params, best_score = walk_forward_optimize(price, strat)
            if best_params:
                best_strats[strat] = best_params
                strat_scores[strat] = best_score
                st.markdown(f"**{strat}** → Best Params: `{best_params}`, Avg OOS Return: `{round(best_score*100, 2)}%`")
                history.setdefault(ticker, {})[strat] = round(best_score, 4)
            else:
                st.warning(f"{strat} failed for {ticker}")

        save_history(history)

        recommended = recommend_strategy(ticker, history)
        if recommended:
            st.info(f"📌 Based on past runs, **{recommended}** has performed best for {ticker}")

        # Backtest
        pf = None
        chosen_strats = []
        if stack_mode == "None":
            if strat_scores:
                top_strat = max(strat_scores.items(), key=lambda x: x[1])
                pf = run_backtest(price, top_strat[0], best_strats[top_strat[0]])
                chosen_strats = [top_strat[0]]
                st.markdown(f"✅ **Backtest: {top_strat[0]}**")
        elif stack_mode == "OR stack":
            if best_strats:
                pf = stack_strategies(price, best_strats)
                chosen_strats = list(best_strats.keys())
                st.markdown("✅ **Backtest: OR-stacked strategies**")
        else:
            if best_strats:
                pf, chosen_strats = stack_by_correlation(price, best_strats, lookback=252, corr_threshold=corr_threshold, metric=corr_metric)
                st.markdown(f"✅ **Backtest: Correlation-based stack** (chosen: {', '.join(chosen_strats)})")

        if pf is None:
            st.warning("No portfolio generated.")
            continue

        pf_dict[ticker] = pf

        # Sentiment overlay
        reddit_sent, karma = get_reddit_sentiment(ticker)
        news_sent = get_news_sentiment(ticker, api_key)
        sentiment_combined = round((reddit_sent + news_sent) / 2, 3)

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price.index, y=price.values, mode='lines', name=f'{ticker} Price'))

        from strategies import build_signals
        entries, exits = build_signals(price, chosen_strats[0], best_strats[chosen_strats[0]])
        fig.add_trace(go.Scatter(x=price.index[entries], y=price[entries], mode='markers',
                                 marker=dict(color='green', size=6), name='Buy'))
        fig.add_trace(go.Scatter(x=price.index[exits], y=price[exits], mode='markers',
                                 marker=dict(color='red', size=6), name='Sell'))

        # Macro overlays
        macro_dict = get_macro_data(start=start_date, end=end_date, selection=macro_selection)

        # 🔍 Debug: Show macro data points
        for name, series in macro_dict.items():
            st.write(f"Macro overlay '{name}': {len(series)} points")

        fig = add_macro_overlays(fig, macro_dict)

        if show_sentiment:
            fig.add_annotation(text=f"🗣️ Sentiment: {sentiment_combined} (Reddit: {round(reddit_sent,3)}, News: {round(news_sent,3)}, Karma: {karma})",
                               xref="paper", yref="paper", x=0.01, y=1.05, showarrow=False, font=dict(size=12))

        fig.update_layout(title=f"{ticker} Strategy Chart", height=540, legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
        if pf.trades.count() == 0:
            st.warning("No trades executed in this strategy.")
        # Stats + Export
        stats = pf.stats()
        if stats is not None and not stats.empty:
            stats_df = stats.to_frame().T
            st.dataframe(stats_df)
        else:
            st.warning("No stats available for this portfolio.")
        csv = pf.trades.records_readable.to_csv(index=False)
        st.download_button("📥 Download Trades CSV", csv, file_name=f"{ticker}_trades.csv", mime="text/csv")

        total_return = stats_df.get("Total Return", pd.Series([np.nan])).iloc[0]
        sharpe_ratio = stats_df.get("Sharpe Ratio", pd.Series([np.nan])).iloc[0]

        comparison_rows.append({
           "Ticker": ticker,
           "Chosen_Strategies": ", ".join(chosen_strats),
           "Total_Return": float(total_return) if pd.notna(total_return) else np.nan,
           "Sharpe_Ratio": float(sharpe_ratio) if pd.notna(sharpe_ratio) else np.nan
        })

    # Comparison chart
    if pf_dict:
        st.subheader("📊 Side-by-Side Cumulative Return Comparison")
        st.plotly_chart(plot_comparison(pf_dict), use_container_width=True)

    # Summary table
    if comparison_rows:
        st.subheader("🧾 Summary: Strategy Choices & Metrics")
        st.dataframe(pd.DataFrame(comparison_rows))