import streamlit as st
import pandas as pd
import yfinance as yf
from core import walk_forward_optimize, run_backtest, stack_strategies
from config import strategy_params
from data import get_price_data
import plotly.graph_objects as go

st.set_page_config(page_title="Multi-Ticker Strategy Lab", layout="wide")
st.title("📊 Multi-Ticker Walk-Forward Strategy Lab")

# --- Sidebar Inputs ---
tickers = st.text_input("Enter tickers (comma-separated)", value="AAPL,MSFT,NVDA").upper().split(",")
start_date = st.date_input("Start date", value=pd.to_datetime("2018-01-01"))
end_date = st.date_input("End date", value=pd.to_datetime("2023-12-31"))
selected_strategies = st.multiselect("Select strategies", list(strategy_params.keys()), default=["MA", "RSI"])
stack_toggle = st.checkbox("Stack strategies for each ticker", value=False)

# --- Run Optimization ---
if st.button("Run Walk-Forward Optimization"):
    results = []
    for ticker in tickers:
        st.subheader(f"📈 {ticker.strip()}")
        price = get_price_data(ticker.strip(), start=start_date, end=end_date)
        if price.empty:
            st.warning(f"No data for {ticker}")
            continue

        best_strats = {}
        for strat in selected_strategies:
            best_params, best_score = walk_forward_optimize(price, strat)
            if best_params:
                best_strats[strat] = best_params
                st.markdown(f"**{strat}** → Best Params: `{best_params}`, Avg OOS Return: `{round(best_score*100, 2)}%`")
            else:
                st.warning(f"{strat} failed for {ticker}")

        # --- Backtest ---
        if stack_toggle and best_strats:
            pf = stack_strategies(price, best_strats)
            st.markdown("✅ **Stacked Strategy Backtest**")
        elif best_strats:
            top_strat = max(best_strats.items(), key=lambda x: walk_forward_optimize(price, x[0])[1])
            pf = run_backtest(price, top_strat[0], top_strat[1])
            st.markdown(f"✅ **Backtest: {top_strat[0]}**")

        # --- Stats + Plot ---
        st.dataframe(pf.stats().to_frame().T)
        st.plotly_chart(pf.plot().figure)

        # --- Export ---
        csv = pf.trades.records_readable.to_csv(index=False)
        st.download_button("📥 Download Trades CSV", csv, file_name=f"{ticker}_trades.csv", mime="text/csv")