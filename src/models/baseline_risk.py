import pandas as pd
import numpy as np
import scipy.stats as stats
from pathlib import Path

# Define project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Constants for baseline
CONFIDENCE_LEVEL = 0.99
ROLLING_WINDOW = 252  # 1 trading year
PORTFOLIO_WEIGHTS = np.array([0.333, 0.333, 0.333]) # Equal weight for SPY, XLF, IYR

def calculate_parametric_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Computes rolling VaR and ES using a historical covariance matrix."""
    print("Computing baseline quantitative risk metrics...")
    
    # Isolate tradable assets (Excluding Lagged_VIX)
    assets = df[['SPY', 'XLF', 'IYR']]
    
    # Calculate rolling mean returns and covariance matrix
    rolling_mean = assets.rolling(window=ROLLING_WINDOW).mean()
    rolling_cov = assets.rolling(window=ROLLING_WINDOW).cov()
    
    risk_metrics = []
    
    # Z-score and PDF for 99% confidence
    z_score = stats.norm.ppf(CONFIDENCE_LEVEL)
    pdf_z = stats.norm.pdf(z_score)
    
    for date in assets.index[ROLLING_WINDOW:]:
        # Extract daily parameters
        mu = rolling_mean.loc[date].values
        cov_matrix = rolling_cov.loc[date].values
        
        # Portfolio expected return: w^T * mu
        port_return = np.dot(PORTFOLIO_WEIGHTS, mu)
        
        # Portfolio variance: w^T * Cov * w
        port_variance = np.dot(PORTFOLIO_WEIGHTS.T, np.dot(cov_matrix, PORTFOLIO_WEIGHTS))
        port_volatility = np.sqrt(port_variance)
        
        # Calculate VaR and ES
        var_99 = port_return - (z_score * port_volatility)
        es_99 = port_return - (port_volatility * (pdf_z / (1 - CONFIDENCE_LEVEL)))
        
        risk_metrics.append({
            'Date': date,
            'Portfolio_Return': port_return,
            'Portfolio_Volatility': port_volatility,
            'VaR_99': var_99,
            'ES_99': es_99
        })
        
    risk_df = pd.DataFrame(risk_metrics).set_index('Date')
    
    # Merge with the original feature matrix (specifically the VIX)
    final_df = df.join(risk_df).dropna()
    
    output_path = PROCESSED_DATA_DIR / "baseline_risk_metrics.csv"
    final_df.to_csv(output_path)
    print(f"Baseline risk metrics successfully written to {output_path}")
    
    return final_df

if __name__ == "__main__":
    # Load processed matrix
    input_path = PROCESSED_DATA_DIR / "processed_returns_matrix.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing processed data at {input_path}. Run ingest_data.py first.")
        
    df = pd.read_csv(input_path, index_col='Date', parse_dates=True)
    
    # Execute risk model
    risk_df = calculate_parametric_risk(df)
    
    print("\nPhase I Complete.")
    print("-" * 40)
    print("Recent Tail Risk Head (VaR vs ES):")
    print(risk_df[['VaR_99', 'ES_99']].tail())