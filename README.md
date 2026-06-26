Market Risk and Portfolio Optimisation Engine
This application is a quantitative risk analytics tool designed for evaluating financial market data. It provides functionality for both single-stock risk analysis and multi-asset portfolio optimisation.
Core Features
Single Stock Analysis
Risk Metrics: Computes Historical VaR, Parametric VaR, and Monte Carlo VaR.  
PY
Expected Shortfall: Calculates the average loss exceeding the Value at Risk threshold.  
PY
Statistical Diagnostics: Performs Jarque-Bera testing to assess normality and identifies potential fat-tail risks.  
PY
Visualization: Provides return distribution plots and scatter plots to visualize VaR breaches over time.  
PY
Portfolio Optimisation
Efficient Frontier: Simulates up to 100,000 portfolio combinations to identify the Efficient Frontier.  
PY
Sharpe Ratio Maximization: Automatically identifies the optimal portfolio allocation that maximizes the Sharpe Ratio.  
PY
Correlation Analysis: Generates a heatmap to evaluate the diversification benefits of selected assets.  
PY
Portfolio Risk: Extends VaR and Expected Shortfall calculations to the optimized portfolio.  
PY
Technical Stack
Framework: Streamlit  
PY
Data Handling: Pandas, NumPy  
PY
Financial Data: yfinance  
PY
Visualization: Matplotlib, Seaborn  
PY
Statistical Analysis: SciPy  
PY
