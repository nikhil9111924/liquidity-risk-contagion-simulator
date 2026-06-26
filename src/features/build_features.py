import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

# Define project architecture paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Target Asset for Liquidity Drain Prediction
TARGET_ASSET = "IYR"

def build_feature_matrix() -> pd.DataFrame:
    """Synthesizes target variables and multi-dimensional features for the ML engine."""
    print("Constructing Machine Learning Feature Matrix...")
    
    # Load Phase I Baseline Data
    input_path = PROCESSED_DATA_DIR / "baseline_risk_metrics.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing baseline data at {input_path}.")
    df = pd.read_csv(input_path, index_col='Date', parse_dates=True)
    
    # 1. Fetch Raw Volume Data for Target Asset
    print(f"Fetching raw volume data for {TARGET_ASSET} to compute Amihud Illiquidity...")
    raw_data = yf.download(TARGET_ASSET, start="2015-01-01", end="2026-01-01", auto_adjust=True)
    
    # Align temporal index
    raw_data = raw_data.loc[df.index]
    
    # Collapse the 2D DataFrames into 1D Series using .squeeze()
    close_series = raw_data['Close'].squeeze()
    volume_series = raw_data['Volume'].squeeze()
    dollar_volume = close_series * volume_series
    
    # 2. Derive Target Variable (Y): Amihud Illiquidity Metric
    # Formula: |Return| / Dollar Volume. Scaled by 10^6 for numerical stability.
    df['Target_Amihud'] = (df[TARGET_ASSET].abs() / dollar_volume) * 1e10
    
    # 3. Feature Engineering (X)
    print("Synthesizing multi-dimensional features...")
    
    # Feature 1: Lagged Returns (Capturing immediate price momentum/panic)
    df['IYR_Ret_Lag1'] = df[TARGET_ASSET].shift(1)
    df['IYR_Ret_Lag3'] = df[TARGET_ASSET].rolling(window=3).sum().shift(1)
    
    # Feature 2: Rolling Volume Momentum (Ratio of short-term to long-term volume)
    # Identifies sudden spikes in trading activity
    vol_5d = raw_data['Volume'].rolling(window=5).mean()
    vol_21d = raw_data['Volume'].rolling(window=21).mean()
    df['Volume_Momentum_5_21'] = (vol_5d / vol_21d).shift(1) 
    
    # Feature 3: Microstructure Volatility (Rolling standard deviation)
    df['Rolling_Vol_21d'] = df[TARGET_ASSET].rolling(window=21).std().shift(1)
    
    # 4. Clean and Finalize the Matrix
    # We strictly use lagged features to completely eliminate look-ahead bias
    feature_columns = [
        'Target_Amihud',          # Target Y
        'SPY', 'XLF',             # Concurrent cross-asset returns
        'Lagged_VIX',             # Concurrent macro-stress indicator
        'IYR_Ret_Lag1', 'IYR_Ret_Lag3', 
        'Volume_Momentum_5_21', 'Rolling_Vol_21d'
    ]
    
    ml_matrix = df[feature_columns].dropna()
    
    # Serialize to disk
    output_path = PROCESSED_DATA_DIR / "ml_feature_matrix.csv"
    ml_matrix.to_csv(output_path)
    print(f"ML Feature Matrix successfully written to {output_path}")
    
    return ml_matrix

if __name__ == "__main__":
    matrix = build_feature_matrix()
    
    print("\nFeature Engineering Complete.")
    print("-" * 40)
    print(f"Final ML Matrix Shape: {matrix.shape}")
    print("\nTarget Variable (Amihud Illiquidity) Statistical Distribution:")
    print(matrix['Target_Amihud'].describe())