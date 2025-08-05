# Configuration file for Quantitative Analysis

# Default tickers to analyze
DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']

# Date range for analysis
START_DATE = '2020-01-01'
END_DATE = '2023-01-01'

# Risk-free rate (annual)
RISK_FREE_RATE = 0.03

# Monte Carlo simulation parameters
MC_DAYS = 252  # 1 year
MC_SIMULATIONS = 1000

# Portfolio optimization parameters
EFFICIENT_FRONTIER_PORTFOLIOS = 10000

# Plotting parameters
FIGURE_SIZE = (12, 8)
DPI = 100

# VaR confidence level
VAR_CONFIDENCE = 0.05  # 95% VaR