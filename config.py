# Configuration file for Quantitative Analysis
from datetime import datetime, timedelta

# Default tickers to analyze
DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']

# Date range for analysis - automatically set to valid dates
END_DATE = datetime.now().strftime('%Y-%m-%d')
START_DATE = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')  # 2 years ago

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