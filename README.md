# Market Risk and Portfolio Optimisation Engine

A quantitative analytics application designed for financial risk assessment and portfolio management. The engine computes various risk metrics, performs statistical diagnostics on asset returns, and identifies optimal portfolio allocations based on Modern Portfolio Theory.

## Features

### Single Stock Analysis
* **Risk Metrics**: Calculates Value at Risk (VaR) using Historical, Parametric, and Monte Carlo methodologies.
* **Expected Shortfall**: Computes the average loss beyond the VaR threshold to quantify tail risk.
* **Statistical Diagnostics**: Employs Jarque-Bera testing to assess return distributions and identifies non-normal, fat-tail behavior.
* **Visualization**: Generates distribution plots and time-series breach scatter plots for performance monitoring.

### Portfolio Optimisation
* **Efficient Frontier**: Simulates large-scale portfolio combinations to visualize the trade-off between risk and return.
* **Optimal Allocation**: Identifies the portfolio maximizing the Sharpe Ratio.
* **Correlation Analysis**: Provides a heatmap of asset returns to assess diversification benefits.
* **Portfolio Risk**: Extends VaR and Expected Shortfall analytics to multi-asset portfolios.

## Technical Stack
* **Framework**: Streamlit
* **Data Analysis**: Pandas, NumPy, SciPy
* **Financial Data**: yfinance
* **Visualization**: Matplotlib, Seaborn

