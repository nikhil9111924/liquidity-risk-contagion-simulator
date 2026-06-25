import yfinance as yf
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Define project root dynamically to ensure path consistency
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Target Asset Matrix
TICKERS = {
    "SPY": "S&P 500 (Broad Market)",
    "XLF": "Financial Select Sector (Banking Contagion)",
    "IYR": "iShares U.S. Real Estate (Target Sector Shock)",
    "^VIX": "CBOE Volatility Index (Macro Stress)"
}

# Define timeline (Capturing major shocks: 2018 Volmageddon, 2020 COVID, 2022 Rate Hikes, 2023 Regional Bank Crisis)
START_DATE = "2015-01-01"
END_DATE = "2026-01-01"

def fetch_market_data() -> pd.DataFrame:
    """Ingests historical adjusted closing prices for the asset matrix."""
    print("Initiating data pipeline...")
    ticker_symbols = list(TICKERS.keys())
    
    # Download data with auto_adjust=True. 
    # This automatically applies dividend and split adjustments to the standard 'Close' column.
    data = yf.download(ticker_symbols, start=START_DATE, end=END_DATE, auto_adjust=True)
    
    # Isolate 'Close' (which is now mathematically equivalent to 'Adj Close')
    adj_close = data['Close']
    
    # Save raw ingestion state
    raw_path = RAW_DATA_DIR / "raw_asset_matrix.csv"
    adj_close.to_csv(raw_path)
    print(f"Raw data successfully written to {raw_path}")
    
    return adj_close

def process_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Aligns dates, handles null values, and computes logarithmic returns."""
    # Forward fill to handle minor temporal discrepancies (e.g., market holidays)
    df = df.ffill().dropna()
    
    # Isolate VIX as it is a level/index, not a return-generating asset
    vix = df['^VIX']
    assets = df[['SPY', 'XLF', 'IYR']]
    
    # Compute daily logarithmic returns for tradable assets
    # Log returns are preferred for quantitative risk models due to time-additivity
    log_returns = np.log(assets / assets.shift(1)).dropna()
    
    # Re-merge VIX (shifted by 1 to align with the return period)
    processed_matrix = log_returns.join(vix.shift(1)).dropna()
    processed_matrix.rename(columns={'^VIX': 'Lagged_VIX'}, inplace=True)
    
    # Save processed state
    processed_path = PROCESSED_DATA_DIR / "processed_returns_matrix.csv"
    processed_matrix.to_csv(processed_path)
    print(f"Processed logarithmic returns written to {processed_path}")
    
    return processed_matrix

if __name__ == "__main__":
    # Pipeline Execution
    raw_df = fetch_market_data()
    processed_df = process_and_normalize(raw_df)
    
    print("\nPipeline Execution Complete.")
    print("-" * 40)
    print("Processed Matrix Data Head:")
    print(processed_df.head())
    print(f"\nMatrix Shape: {processed_df.shape} | Null Values: {processed_df.isnull().sum().sum()}")