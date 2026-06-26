# AI-Driven Liquidity Risk Infrastructure & Neural Network Contagion Simulator
**Project Proposal & Technical Architecture** **Author:** Nikhil Sivakumar  

---

## Abstract
Most traditional financial risk models assume that an institution can always sell its assets at current market prices. In reality, during a market panic, buyers disappear. Trying to offload massive positions causes prices to crash further, trapping the institution in a "fire sale" loop—a dynamic that drove the recent collapses of several major regional banks. 

This project outlines an AI-powered risk platform designed to fix this blind spot. It uses machine learning to predict the true, hidden costs of liquidating assets during a crisis. Additionally, it features an interactive network simulator to map how a shock in one sector (such as commercial real estate) can trigger a chain reaction that spreads across an entire global portfolio. Ultimately, this tool equips executives with the predictive insights needed to prevent liquidity death spirals before they begin.

## 1. Problem Statement
Standard risk calculations, such as Value at Risk (VaR), treat market liquidity as a static variable. During periods of institutional panic, liquidity dries up non-linearly, causing actual execution costs to exponentially exceed theoretical estimates. 

Furthermore, classical asset correlation matrices fail to capture the cascading, structural failures that occur when financial institutions are forced into simultaneous fire sales. To accurately evaluate portfolio resilience, risk systems must transition from static statistical models to predictive, data-driven AI systems capable of forecasting real-time market impact and network-level failure propagation.

## 2. Project Objectives
* **Develop a Liquidity Regressor Pipeline:** Implement and train a tree-based machine learning model to predict bid-ask spread widening and slippage costs based on multi-dimensional market microstructure features.
* **Architect a Neural Contagion Model:** Deploy Graph Convolutional Networks (GCNs) to learn spatial representations of cross-asset dependencies and simulate how systemic shocks propagate across a global portfolio.
* **Incorporate Predictive Intelligence into Asset Allocation:** Utilize AI-driven risk outputs to dynamically optimize portfolio weights to minimize expected tail losses during macro shocks.

## 3. Technical Architecture & Machine Learning Scope

### 3.1 Machine Learning Execution Impact Model
Instead of relying on rigid mathematical heuristics for liquidation penalties, this engine introduces a predictive machine learning pipeline.
* **Feature Engineering:** Input matrices (X) will consist of time-series features including the Amihud Illiquidity Ratio, rolling volume standard deviation, order book imbalance metrics, and macroeconomic volatility indicators (VIX).
* **Model Selection:** A gradient-boosted ensemble model (XGBoost/LightGBM) will be optimized using hyperparameter tuning (GridSearchCV) to map these non-linear features to the target variable (Y): expected liquidation slippage cost.
* **Evaluation Metrics:** Model performance will be evaluated using Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE). *(Note: Adapted to Mean Absolute Error for zero-bound financial stability)*.

### 3.2 Deep Learning Graph Contagion Simulator
To model systemic risk as a geometric problem, the portfolio is treated as a force-directed graph $\mathcal{G}=(\mathcal{V},\mathcal{E})$, where vertices $\mathcal{V}$ represent asset classes and edges $\mathcal{E}$ represent dynamic correlation weights.
* **Architecture:** A Graph Convolutional Network (GCN) / NetworkX topology will be implemented to perform node classification and regression tasks, learning how an exogenous shock applied to a specific node alters the hidden states of adjacent asset nodes.
* **Training Protocol:** The network will be trained on historical financial crisis data to capture asymmetric tail dependencies that traditional linear correlation models overlook.

### 3.3 Optimization Backtesting Frame
The predictive outputs of the ML and GCN modules will feed into an active portfolio optimization framework, dynamically computing an AI-adjusted Expected Shortfall (ES) to guide capital allocation decisions.

## 4. Expected Portfolio & Resume Impact
This architecture demonstrates an advanced synthesis of financial engineering and machine learning engineering. The inclusion of end-to-end model training pipelines, feature selection methods, and deep learning network topologies provides explicit technical validation for quantitative research and business analytics positions at institutions like Goldman Sachs and JPMorgan Chase.