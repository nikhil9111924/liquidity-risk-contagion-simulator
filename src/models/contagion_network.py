import pandas as pd
import numpy as np
import networkx as nx
import xgboost as xgb
import pickle
from pathlib import Path

# Architecture Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

# Contagion Parameters
CORRELATION_THRESHOLD = 0.40  # Minimum correlation to instantiate a contagion edge
SHOCK_NODE = "IYR"
SHOCK_MAGNITUDE = -0.10  # A sudden 10% crash in Commercial Real Estate

def build_topology_and_simulate():
    print("Initializing Network Contagion Architecture...")
    
    # 1. Load the Data & ML Model
    ml_data_path = PROCESSED_DATA_DIR / "ml_feature_matrix.csv"
    ml_df = pd.read_csv(ml_data_path, index_col='Date', parse_dates=True)
    
    # Load Baseline matrix for the topological correlations
    baseline_path = PROCESSED_DATA_DIR / "baseline_risk_metrics.csv"
    baseline_df = pd.read_csv(baseline_path, index_col='Date', parse_dates=True)
    
    model_path = MODEL_DIR / "xgboost_liquidity_model.pkl"
    with open(model_path, 'rb') as f:
        liquidity_model = pickle.load(f)
        
    # 2. Construct the Asset Topology (Graph)
    print("Constructing topological edge weights via historical correlation...")
    assets = ['SPY', 'XLF', 'IYR']
    
    # Use the baseline_df to calculate correlation weights
    correlation_matrix = baseline_df[assets].corr()
    
    G = nx.Graph()
    for asset in assets:
        G.add_node(asset, shock_state=0.0, liquidity_drain=0.0)
        
    # Instantiate edges based on correlation threshold
    for i in range(len(assets)):
        for j in range(i+1, len(assets)):
            weight = correlation_matrix.iloc[i, j]
            if abs(weight) >= CORRELATION_THRESHOLD:
                G.add_edge(assets[i], assets[j], weight=weight)
                print(f"Edge Formed: {assets[i]} <--> {assets[j]} (Weight: {weight:.4f})")

    # 3. Trigger Exogenous Shock
    print(f"\nTriggering Exogenous Shock: {SHOCK_NODE} crashes by {SHOCK_MAGNITUDE:.2%}")
    G.nodes[SHOCK_NODE]['shock_state'] = SHOCK_MAGNITUDE
    
    # 4. Predict Liquidity Drain (ML Inference)
    # Use the ml_df for the synthetic worst-case row
    latest_data = ml_df.iloc[-1].copy()
    latest_data['Rolling_Vol_21d'] *= 3.0  
    latest_data['IYR_Ret_Lag1'] = SHOCK_MAGNITUDE
    
    # Prepare inference vector (must match Phase II feature order exactly)
    feature_cols = ['SPY', 'XLF', 'Lagged_VIX', 'IYR_Ret_Lag1', 'IYR_Ret_Lag3', 'Volume_Momentum_5_21', 'Rolling_Vol_21d']
    inference_vector = latest_data[feature_cols].values.reshape(1, -1)
    
    # The ML model calculates the exact Amihud execution penalty
    predicted_drain = liquidity_model.predict(inference_vector)[0]
    G.nodes[SHOCK_NODE]['liquidity_drain'] = predicted_drain
    
    print(f"ML Execution Engine predicts a systemic liquidity drain factor of: {predicted_drain:.6f}")
    
    # 5. Shock Propagation Algorithm
    print("\nExecuting Network Shock Cascade...")
    propagation_log = []
    
    # Distribute fractional shocks to adjacent nodes
    for neighbor in G.neighbors(SHOCK_NODE):
        edge_weight = G[SHOCK_NODE][neighbor]['weight']
        
        # The equation for contagion: Shock * Correlation * (1 + Liquidity Drain)
        contagion_impact = SHOCK_MAGNITUDE * edge_weight * (1 + predicted_drain)
        
        G.nodes[neighbor]['shock_state'] = contagion_impact
        propagation_log.append(f"-> Node {neighbor} infected: Absorbed a {contagion_impact:.2%} cascading loss.")

    for log in propagation_log:
        print(log)
        
    print("\nTerminal State of Portfolio Nodes:")
    for node, data in G.nodes(data=True):
        print(f"[{node}] Total Impact: {data['shock_state']:.2%}")

if __name__ == "__main__":
    build_topology_and_simulate()