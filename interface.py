import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
from bar_permute import get_permutation
import plotly.graph_objects as go

# Streamlit App Setup
st.set_page_config(page_title="Strategy Tester", layout="wide")
st.title("Donchian Strategy In-Sample MCPT Testing")

# File Uploaders
strategy_file = st.sidebar.file_uploader("Upload Strategy Python File", type=["py"])
data_file = st.sidebar.file_uploader("Upload BTC-USD Data (PQ)", type=["pq"])

# Load Data
@st.cache_data
def load_data(file):
    df = pd.read_parquet(file)
    df.index = df.index.tz_localize(None)
    return df

if data_file is not None:
    df = load_data(data_file)
    df = df[(df.index.year >= 2024) & (df.index.year < 2025)]
    st.sidebar.success("Data Loaded Successfully")
else:
    st.sidebar.warning("Please upload BTC-USD data")
    st.stop()

# Strategy Execution
if strategy_file is not None:
    # Save and import strategy
    with open("strategy.py", "wb") as f:
        f.write(strategy_file.getbuffer())
    
    spec = importlib.util.spec_from_file_location("strategy", "strategy.py")
    strategy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy)
    
    st.header("BTC-USD Price Action")
    
    # Candlestick Chart
    fig_candle = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df[('Open', 'BTC-USD')],
        high=df[('High', 'BTC-USD')],
        low=df[('Low', 'BTC-USD')],
        close=df[('Close', 'BTC-USD')]
    )])

    fig_candle.update_layout(
        title='BTC-USD Price Action',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        dragmode='pan'  # Set default drag mode to pan
    )

    st.plotly_chart(fig_candle, use_container_width=True)
    
    # Run In-Sample Backtest
    st.header("In-Sample Backtest Results")
    
    best_lookback, best_real_pf = strategy.optimize_donchian(df)
    signal = strategy.donchian_breakout(df, best_lookback)
    
    df['r'] = np.log(df[('Close', 'BTC-USD')]).diff().shift(-1)
    df['donch_r'] = df['r'] * signal
    
    
    # Plot Cumulative Returns
    fig, ax = plt.subplots(figsize=(12, 6))
    df['donch_r'].cumsum().plot(ax=ax, color='red')
    ax.set_title(f"In-Sample Donchian Breakout (Lookback: {best_lookback})", fontsize=14)
    ax.set_ylabel('Cumulative Log Return', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

    # Run In-Sample MCPT
    st.header("In-Sample MCPT Results")
    n_permutations = st.sidebar.slider("Number of Permutations", 50, 1000, 100)
    
    perm_better_count = 1
    permuted_pfs = []
    
    progress_text = st.empty()  # Text to show current iteration
    progress_bar = st.progress(0.0)  # Progress bar
    for i in range(1, n_permutations):
        progress = i / (n_permutations - 1)

        # Update progress bar and text
        progress_bar.progress(progress)
        progress_text.text(f"Running permutation {i}/{n_permutations-1}")
        
        perm_df = get_permutation(df)
        _, best_perm_pf = strategy.optimize_donchian(perm_df)
        
        if best_perm_pf >= best_real_pf:
            perm_better_count += 1
            
        permuted_pfs.append(best_perm_pf)
    
    p_value = perm_better_count / n_permutations
    progress_text.empty()

    # Display Results
    col1, col2 = st.columns([3, 1])
    with col1:
        fig_hist, ax_hist = plt.subplots(figsize=(12, 6))
        pd.Series(permuted_pfs).hist(ax=ax_hist, bins=30, color='blue', alpha=0.7)
        ax_hist.axvline(best_real_pf, color='red', linestyle='--', linewidth=2)
        ax_hist.set_title(f"In-sample MCPT (p-value={p_value:.4f})", fontsize=14)
        ax_hist.set_xlabel("Profit Factor")
        ax_hist.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig_hist)
    
    with col2:
        st.markdown("<h2 style='text-align: center;'>Results</h2>", 
                   unsafe_allow_html=True)
        st.metric("Real Profit Factor", f"{best_real_pf:.4f}")
        st.metric("P-Value", f"{p_value:.4f}", 
                 delta_color="off",
                 help="Probability of achieving this result by chance")
        
        # if p_value < 0.05:
        #     st.success("Statistically Significant (p < 0.05)")
        # else:
        #     st.warning("Not Statistically Significant")

else:
    st.sidebar.warning("Please upload strategy file")