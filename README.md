# Quantitative Finance Analysis Tool

A comprehensive Python-based tool for quantitative finance analysis, featuring risk metrics, portfolio analysis, Monte Carlo simulations, efficient frontier, and more. Supports both command-line and graphical user interfaces.

## Features

- **Stock Data Loading**: Fetch historical price data from Yahoo Finance for multiple tickers
- **Data Validation**: Automatic validation and cleaning of tickers and date ranges
- **Risk & Return Metrics**: Calculate annualized returns, volatility, Sharpe ratio, Value at Risk (VaR), and maximum drawdown
- **Portfolio Analysis**: Analyze portfolio performance with custom or equal weights
- **Monte Carlo Simulation**: Simulate thousands of random portfolios or individual stock paths for risk/return forecasting
- **Correlation Analysis**: Visualize asset correlations with heatmaps
- **Efficient Frontier**: Find optimal portfolio allocations and visualize the risk-return tradeoff
- **Multiple Interfaces**: Command-line, GUI, and programmatic access
- **Export Results**: Save results to Excel or CSV, including all metrics and analysis reports
- **Comprehensive Logging & Error Handling**: Detailed feedback and troubleshooting

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Tool

**Option A: Graphical User Interface (GUI)**
```bash
python main_app.py
```

**Option B: Command-Line Interface (CLI)**
```bash
python main_app.py --tickers "AAPL,MSFT,GOOGL" --analysis all --start "2022-01-01" --end "2024-01-01"
```

**Option C: Use Provided Batch Files (Windows)**
```bash
run_gui.bat
run_analysis.bat
run_monte_carlo.bat
run_diagnostic.bat
```

## Command-Line Options

| Option           | Description                                                      | Default                        |
|------------------|------------------------------------------------------------------|--------------------------------|
| --tickers        | Comma-separated list of stock tickers                            | AAPL,MSFT                      |
| --start          | Start date (YYYY-MM-DD)                                          | 2 years ago (see config.py)    |
| --end            | End date (YYYY-MM-DD)                                            | today (see config.py)          |
| --analysis       | Type of analysis: basic, portfolio, montecarlo, correlation, all | all                            |
| --export         | Export results to file (CSV or Excel)                            | None                           |
| --simulations    | Number of Monte Carlo simulations                                | 1000                           |
| --days           | Number of days to forecast (Monte Carlo)                         | 252                            |
| --all-stocks     | Run Monte Carlo for all stocks individually                      | False                          |
| --gui            | Force GUI mode even with other arguments                         | False                          |

See all options with:
```bash
python main_app.py --help
```

## GUI Usage
- Enter tickers, date range, and simulation parameters in the input fields
- Use buttons to load data, calculate metrics, show correlation, run Monte Carlo, efficient frontier, or full analysis
- Results and charts are displayed in tabs
- Export results to Excel or CSV with the "Export Results" button

## Example Workflows

**Basic Analysis:**
```bash
python main_app.py --tickers "AAPL" --analysis basic
```

**Portfolio Analysis:**
```bash
python main_app.py --tickers "AAPL,MSFT,GOOGL,TSLA" --analysis portfolio
```

**Monte Carlo Simulation:**
```bash
python main_app.py --tickers "TSLA" --analysis montecarlo --simulations 2000 --days 365
```

**Full Analysis with Export:**
```bash
python main_app.py --tickers "AAPL,MSFT,GOOGL,TSLA,AMZN" --analysis all --export full_analysis.xlsx
```

## Configuration

Edit `config.py` to customize default settings:
- Default tickers and date range
- Risk-free rate for Sharpe ratio
- Monte Carlo and efficient frontier parameters
- Plotting options

## Output Files
- **Excel Export**: Results with multiple sheets (returns, metrics, correlation, analysis report)
- **CSV Export**: Simple returns data
- **Charts**: Displayed interactively (not saved by default)

## Key Metrics Calculated
- **Annual Return**: Annualized average return
- **Annual Volatility**: Annualized standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted return measure
- **Value at Risk (VaR)**: Potential loss at 95% confidence level
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Portfolio Metrics**: Combined portfolio statistics
- **Correlation Matrix**: Asset correlation analysis

## Troubleshooting

- **Everything is NaN**: Run `python diagnostic.py`, check internet connection, verify tickers, adjust date range, or fix matplotlib issues with `python fix_matplotlib.py`.
- **No module named 'yfinance'**: Install dependencies with `pip install -r requirements.txt`.
- **Invalid ticker symbols**: Use correct, up-to-date symbols (e.g., AAPL, MSFT).
- **No data for date range**: Use valid market dates, avoid weekends/holidays, or try a longer range.
- **Matplotlib backend error**: Run `python fix_matplotlib.py`.

## Support

If you encounter issues:
1. Run the diagnostic: `python diagnostic.py`
2. Check the troubleshooting section above
3. Verify your internet connection
4. Ensure all dependencies are installed
5. Try with different ticker symbols or date ranges

For matplotlib issues specifically:
```bash
python fix_matplotlib.py
```

## For Developers: Programmatic Usage

```python
from quant_analysis import QuantitativeAnalysis

qa = QuantitativeAnalysis()
data = qa.load_stock_data(['AAPL', 'MSFT'], '2022-01-01', '2024-01-01')
returns = qa.calculate_returns()
metrics = qa.calculate_risk_return_metrics()
portfolio_results = qa.portfolio_analysis()
mc_results = qa.monte_carlo_simulation(days=252, simulations=1000)
correlation = qa.correlation_analysis()
ef_results = qa.efficient_frontier()
```

## Requirements

See `requirements.txt` for a complete list:
- yfinance >= 0.1.87
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scipy >= 1.7.0
- statsmodels >= 0.12.0

---

This tool provides professional-grade quantitative analysis capabilities suitable for education, research, and investment analysis.