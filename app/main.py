import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import pickle
import plotly.graph_objects as go
from pathlib import Path

# --- Architecture Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

# --- Page Configuration ---
st.set_page_config(page_title="Liquidity Risk & Contagion Simulator", layout="wide")
st.title("AI-Driven Liquidity Risk & Contagion Simulator")
st.markdown("Interactive network simulator mapping exogenous shocks and ML-predicted liquidity execution penalties across a multi-asset portfolio.")

# --- Data & Model Loading ---
@st.cache_data
def load_data():
    ml_df = pd.read_csv(PROCESSED_DATA_DIR / "ml_feature_matrix.csv", index_col='Date', parse_dates=True)
    baseline_df = pd.read_csv(PROCESSED_DATA_DIR / "baseline_risk_metrics.csv", index_col='Date', parse_dates=True)
    return ml_df, baseline_df

@st.cache_resource
def load_model():
    with open(MODEL_DIR / "xgboost_liquidity_model.pkl", 'rb') as f:
        return pickle.load(f)

ml_df, baseline_df = load_data()
liquidity_model = load_model()

# --- Sidebar Controls ---
st.sidebar.header("Macroeconomic Stress Parameters")
shock_node = st.sidebar.selectbox("Exogenous Shock Origin", ["IYR", "XLF", "SPY"], index=0)
shock_magnitude = st.sidebar.slider("Shock Magnitude (Crash %)", min_value=-30.0, max_value=0.0, value=-10.0, step=1.0) / 100.0
volatility_multiplier = st.sidebar.slider("Market Volatility Multiplier", min_value=1.0, max_value=5.0, value=3.0, step=0.5)

# --- Simulation Engine ---
assets = ['SPY', 'XLF', 'IYR']
correlation_matrix = baseline_df[assets].corr()

# 1. Topology
G = nx.Graph()
for asset in assets:
    G.add_node(asset, shock=0.0)

for i in range(len(assets)):
    for j in range(i+1, len(assets)):
        weight = correlation_matrix.iloc[i, j]
        if abs(weight) >= 0.40:
            G.add_edge(assets[i], assets[j], weight=weight)

# 2. ML Inference for Liquidity Drain
latest_data = ml_df.iloc[-1].copy()
latest_data['Rolling_Vol_21d'] *= volatility_multiplier
# Map the shock to the appropriate lag feature if it's IYR, otherwise simulate a general drop
if shock_node == "IYR":
    latest_data['IYR_Ret_Lag1'] = shock_magnitude

feature_cols = ['SPY', 'XLF', 'Lagged_VIX', 'IYR_Ret_Lag1', 'IYR_Ret_Lag3', 'Volume_Momentum_5_21', 'Rolling_Vol_21d']
inference_vector = latest_data[feature_cols].values.reshape(1, -1)
predicted_drain = liquidity_model.predict(inference_vector)[0]

# 3. Network Propagation
G.nodes[shock_node]['shock'] = shock_magnitude
for neighbor in G.neighbors(shock_node):
    edge_weight = G[shock_node][neighbor]['weight']
    contagion_impact = shock_magnitude * edge_weight * (1 + predicted_drain)
    G.nodes[neighbor]['shock'] = contagion_impact

# --- Executive Dashboard Rendering ---
st.subheader("Systemic Impact Analysis")
col1, col2, col3 = st.columns(3)
col1.metric("ML Execution Penalty (Slippage)", f"{predicted_drain:.2%}")
col2.metric(f"Direct Shock ({shock_node})", f"{shock_magnitude:.2%}")
col3.metric("Broad Market Impact (SPY)", f"{G.nodes['SPY']['shock']:.2%}" if shock_node != 'SPY' else f"{shock_magnitude:.2%}")

# --- Visualizer: Bar Chart of Total Impact ---
st.markdown("### Portfolio Contagion Distribution")
impact_data = {node: data['shock'] for node, data in G.nodes(data=True)}
impact_df = pd.DataFrame(list(impact_data.items()), columns=['Asset', 'Total Impact'])

fig = go.Figure(go.Bar(
    x=impact_df['Asset'],
    y=impact_df['Total Impact'],
    marker_color=['red' if x < -0.05 else 'orange' for x in impact_df['Total Impact']]
))
fig.update_layout(yaxis_title="Decline Percentage", xaxis_title="Asset Class", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)