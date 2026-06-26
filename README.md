# AI-Driven Liquidity Risk & Contagion Simulator

> An end-to-end quantitative risk platform that uses machine learning to predict hidden liquidation costs and simulates how financial shocks cascade across a multi-asset portfolio in real time.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://liquidity-risk-contagion-simulator-9inmx6jyiwdjfzbyknfogz.streamlit.app/)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)
![NetworkX](https://img.shields.io/badge/Graph-NetworkX-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**[▶ Launch the Live Simulator](https://liquidity-risk-contagion-simulator-9inmx6jyiwdjfzbyknfogz.streamlit.app/)**

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technical Architecture](#technical-architecture)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Installation](#local-installation)
  - [Docker](#docker)
  - [GitHub Codespaces](#github-codespaces)
- [Pipeline Execution Guide](#pipeline-execution-guide)
- [How the Simulator Works](#how-the-simulator-works)
- [Model Performance](#model-performance)
- [Data Sources](#data-sources)
- [Author](#author)

---

## Problem Statement

Standard risk models such as Value at Risk (VaR) treat market liquidity as a static constant. In reality, during periods of institutional panic, liquidity dries up **non-linearly** — actual execution costs can exponentially exceed theoretical estimates. The 2023 regional bank crisis demonstrated how a localized shock in commercial real estate can trigger cascading "fire sale" loops across an entire financial system.

Traditional correlation matrices fail to capture these cascading, structural failures. Financial risk systems must move beyond static statistics toward **predictive, data-driven AI** capable of forecasting real-time market impact and network-level failure propagation.

## Project Overview

This platform addresses the liquidity blind spot through two core AI components:

1. **Machine Learning Execution Engine** — A gradient-boosted ensemble (XGBoost) trained on historical market microstructure data to predict the Amihud Illiquidity Metric, forecasting the true cost of liquidating assets during a crisis.

2. **Neural Contagion Simulator** — A force-directed graph topology (NetworkX) that maps cross-asset dependencies and simulates how an exogenous shock propagates through a portfolio, amplified by the ML-predicted liquidity drain.

The result is an interactive, executive-facing dashboard that lets users apply macroeconomic stress scenarios and observe contagion dynamics in real time.

---

## Key Features

| Feature | Description |
|---|---|
| **AI Liquidity Prediction** | XGBoost regressor forecasts execution slippage from multi-dimensional market microstructure features |
| **Contagion Network** | Graph-based shock propagation engine distributes fractional losses across correlated asset nodes |
| **Interactive Stress Testing** | Sidebar controls for shock origin, crash magnitude, and volatility multipliers |
| **Executive Dashboard** | Real-time KPI tiles and Plotly visualizations rendering portfolio impact distributions |
| **End-to-End Pipeline** | Automated data ingestion → feature engineering → model training → live inference |
| **Cloud Deployed** | Live on Streamlit Community Cloud with Docker containerization support |

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA ARCHITECTURE (Phase I)                    │
│                                                                     │
│  yfinance API ──► Raw Ingestion ──► Log Returns ──► Baseline VaR/ES│
│  (SPY, XLF,       (data/raw/)       (Temporal        (Parametric    │
│   IYR, ^VIX)                         Alignment)       99% CI)       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   ML EXECUTION ENGINE (Phase II)                    │
│                                                                     │
│  Feature Matrix (X)              Target Variable (Y)                │
│  ├─ Lagged Returns               Amihud Illiquidity Metric          │
│  ├─ Volume Momentum              |Rₜ| / Dollar Volume              │
│  ├─ Rolling Volatility                                              │
│  └─ Lagged VIX                  XGBoost Regressor ──► .pkl Model    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│               NETWORK CONTAGION ENGINE (Phase III)                   │
│                                                                     │
│  G = (V, E)                                                         │
│  Nodes: Asset Classes          Edges: Correlation > 0.40            │
│                                                                     │
│  Shock(Node_A) ──► Contagion = Shock × Corr × (1 + ML_Drain)       │
│                         ──► Adjacent node states updated            │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STREAMLIT EXECUTIVE DASHBOARD (Phase IV)                │
│                                                                     │
│  Sidebar Controls ──► ML Inference ──► Network Cascade ──► Plotly   │
│  (Shock Origin,       (Real-time       (State             (Bar      │
│   Magnitude,           Prediction)      Propagation)       Charts)  │
│   Vol Multiplier)                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Core Mathematical Formulations

**Amihud Illiquidity Metric** (ML Target Variable):

$$Illiquidity_t = \frac{|R_t|}{DollarVolume_t} \times 10^{10}$$

**Parametric Value at Risk** (Baseline Benchmark):

$$VaR_{99} = \mu_p - z_{0.99} \cdot \sigma_p$$

**Contagion Propagation Equation**:

$$Contagion_{i \to j} = Shock_i \times \rho_{ij} \times (1 + \hat{D}_{ML})$$

Where $\hat{D}_{ML}$ is the XGBoost-predicted liquidity drain factor and $\rho_{ij}$ is the historical correlation edge weight.

---

## Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python 3.11 |
| **Data Ingestion** | yfinance, pandas, NumPy |
| **Statistical Modeling** | SciPy (parametric VaR / Expected Shortfall) |
| **Machine Learning** | XGBoost, scikit-learn |
| **Graph Topology** | NetworkX |
| **Deep Learning (Framework)** | PyTorch, PyTorch Geometric |
| **Frontend / Dashboard** | Streamlit |
| **Visualization** | Plotly |
| **Containerization** | Docker |
| **Deployment** | Streamlit Community Cloud |
| **Dev Environment** | VS Code Dev Containers / GitHub Codespaces |

---

## Repository Structure

```
liquidity-risk-contagion-simulator/
│
├── app/
│   └── main.py                          # Streamlit dashboard & live inference engine
│
├── src/
│   ├── data/
│   │   └── ingest_data.py               # Phase I: yfinance data pipeline & log return computation
│   ├── features/
│   │   └── build_features.py            # Phase II: Amihud target derivation & feature engineering
│   ├── models/
│   │   ├── baseline_risk.py             # Phase I: Parametric VaR & Expected Shortfall computation
│   │   ├── train_xgboost.py             # Phase II: XGBoost training, evaluation & serialization
│   │   └── contagion_network.py         # Phase III: NetworkX graph construction & shock cascade
│   └── visualization/                   # (Reserved for standalone plot utilities)
│
├── models/
│   └── xgboost_liquidity_model.pkl      # Serialized XGBoost regression weights
│
├── data/
│   ├── raw/
│   │   └── raw_asset_matrix.csv         # Raw adjusted close prices from yfinance
│   └── processed/
│       ├── processed_returns_matrix.csv  # Cleaned logarithmic returns with lagged VIX
│       ├── baseline_risk_metrics.csv     # Returns + rolling VaR/ES/Volatility
│       └── ml_feature_matrix.csv         # Final feature matrix (X) with Amihud target (Y)
│
├── .devcontainer/
│   └── devcontainer.json                # GitHub Codespaces / VS Code Dev Container config
│
├── Dockerfile                           # Production container (Python 3.11-slim)
├── requirements.txt                     # Python dependencies grouped by phase
├── packages.txt                         # System-level packages for Streamlit Cloud (libgomp1)
├── PROPOSAL.md                          # Technical project proposal & ML architecture spec
├── ROADMAP.md                           # Phased implementation roadmap with stage-gates
└── README.md                            # This file
```

---

## Getting Started

### Prerequisites

- **Python** ≥ 3.11
- **pip** (Python package manager)
- **Git**
- **Docker** (optional, for containerized deployment)

### Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/nikhil9111924/liquidity-risk-contagion-simulator.git
cd liquidity-risk-contagion-simulator

# 2. Create and activate a virtual environment
python -m venv myenv
source myenv/bin/activate        # macOS / Linux
myenv\Scripts\activate           # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Launch the Streamlit dashboard
streamlit run app/main.py
```

The application will open at `http://localhost:8501`.

### Docker

```bash
# Build the Docker image
docker build -t liquidity-simulator .

# Run the container
docker run -p 8501:8501 liquidity-simulator
```

Access the dashboard at `http://localhost:8501`.

### GitHub Codespaces

This repository includes a `.devcontainer` configuration for one-click setup:

1. Click **"Code" → "Codespaces" → "Create codespace on main"** on the GitHub repo page.
2. The environment will automatically install all dependencies and launch the Streamlit server.
3. The application preview will open on port `8501`.

---

## Pipeline Execution Guide

If you need to rebuild the data pipeline and retrain the model from scratch, execute the scripts sequentially:

```bash
# Phase I — Data Ingestion & Baseline Risk Computation
python src/data/ingest_data.py
python src/models/baseline_risk.py

# Phase II — Feature Engineering & ML Model Training
python src/features/build_features.py
python src/models/train_xgboost.py

# Phase III — Contagion Network Simulation (standalone test)
python src/models/contagion_network.py

# Phase IV — Launch Interactive Dashboard
streamlit run app/main.py
```

> **Note:** The processed data and trained model are pre-committed in the repository, so you can skip directly to launching the dashboard if you do not need to refresh the data.

---

## How the Simulator Works

1. **Select Stress Parameters** — Use the sidebar to choose a shock origin node (SPY, XLF, or IYR), set the crash magnitude (up to −30%), and adjust the market volatility multiplier.

2. **ML Inference** — The XGBoost model takes the latest market features (augmented by your stress parameters) and predicts the expected liquidity drain factor (Amihud Illiquidity).

3. **Network Propagation** — The shock is applied to the selected node. The contagion equation distributes fractional impacts to all adjacent nodes in the correlation-weighted graph.

4. **Dashboard Rendering** — KPI tiles display the ML-predicted execution penalty, the direct shock value, and the broad market impact. A color-coded bar chart visualizes the contagion distribution across all portfolio assets.

---

## Model Performance

The XGBoost regressor is trained on an 80/20 chronological split to prevent look-ahead bias.

| Metric | Description |
|---|---|
| **RMSE** | Root Mean Squared Error on test set |
| **MAE** | Mean Absolute Error on test set |

**Hyperparameters:**
- `n_estimators`: 100
- `learning_rate`: 0.05
- `max_depth`: 4
- `subsample`: 0.8
- `colsample_bytree`: 0.8

---

## Data Sources

| Asset / Index | Ticker | Role in Portfolio |
|---|---|---|
| S&P 500 ETF | `SPY` | Broad market benchmark |
| Financial Select Sector SPDR | `XLF` | Banking sector contagion vector |
| iShares U.S. Real Estate ETF | `IYR` | Primary shock origin (CRE exposure) |
| CBOE Volatility Index | `^VIX` | Macroeconomic stress indicator |

**Time Range:** January 2015 – January 2026  
**Data Provider:** [Yahoo Finance](https://finance.yahoo.com/) via `yfinance`

This window captures major market dislocations including: 2018 Volmageddon, 2020 COVID crash, 2022 rate hike cycle, and the 2023 regional banking crisis.

---

## Author

**Nikhil Sivakumar**

---
