# AI-Driven Liquidity Risk & Contagion Simulator

![Live Application Demo](link-to-your-deployed-app-or-gif)

## Executive Summary
An end-to-end quantitative infrastructure designed to forecast execution slippage using a gradient-boosted ensemble (XGBoost) and map multi-asset systemic shock propagation via Graph Convolutional Networks and force-directed topologies (NetworkX, PyTorch).

## 1. Machine Learning Execution Engine (Phase II)
Traditional risk models treat liquidity as static. This engine predicts the **Amihud Illiquidity Metric** dynamically to calculate execution penalties during fire sales.

$$Illiquidity = \frac{|R_t|}{V_t}$$

## 2. Geometric Shock Cascade (Phase III)
Systemic risk is mapped as a force-directed graph $\mathcal{G}=(\mathcal{V},\mathcal{E})$. When an exogenous shock is applied to a node, the fractional contagion is calculated via historical correlation and the AI-predicted execution penalty:

$$Contagion = Shock \times Correlation \times (1 + \text{Predicted Drain})$$

## Repository Architecture
- `/src`: Data pipelines, ML feature engineering, and network topologies.
- `/models`: Serialized XGBoost regression weights.
- `/app`: Streamlit executive dashboard.

## Live Deployment
[Access the Live Simulator Here](link-to-your-deployed-url)