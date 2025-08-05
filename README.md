# Quantitative Finance Analysis Tool

A comprehensive Python-based quantitative finance analysis tool for stock market analysis, portfolio optimization, and risk management with both GUI and CLI interfaces.

## Features

- **Stock Data Analysis**: Download and analyze historical stock data
- **Risk & Return Metrics**: Calculate annualized returns, volatility, Sharpe ratio, VaR, and maximum drawdown
- **Portfolio Analysis**: Portfolio optimization with efficient frontier analysis
- **Monte Carlo Simulation**: Forecast future price movements using Monte Carlo methods
- **Correlation Analysis**: Visualize correlations between different assets
- **Unified Interface**: Single application with both GUI and CLI modes

## Prerequisites

Before running the tool, ensure you have:
- Python 3.7 or higher installed
- Internet connection (for downloading stock data)
- Windows, macOS, or Linux operating system

## Installation & Setup

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation
```bash
python -c "import yfinance, pandas, numpy, matplotlib; print('All dependencies installed successfully!')"
```

## How to Run

### Method 1: Quick Start with Batch Files (Windows)

#### Option A: GUI Mode (Interactive)
```bash
run_gui.bat
```
Opens the graphical user interface for interactive analysis.

#### Option B: Full CLI Analysis
```bash
run_analysis.bat
```
Runs comprehensive analysis on AAPL, MSFT, GOOGL, and TSLA stocks.

#### Option C: Quick Analysis
```bash
run_quick_analysis.bat
```
Runs basic metrics analysis only.

#### Option D: Monte Carlo Focus
```bash
run_monte_carlo.bat
```
Runs Monte Carlo simulation for all specified stocks.

### Method 2: Direct Python Execution

#### Default Mode (GUI)
```bash
python main_app.py
```
Starts in GUI mode when no arguments are provided.

#### Force GUI Mode
```bash
python main_app.py --gui
```

#### CLI Mode - Basic Analysis
```bash
python main_app.py --tickers "AAPL,MSFT,GOOGL" --analysis basic
```

#### CLI Mode - Full Analysis
```bash
python main_app.py --tickers "AAPL,MSFT,GOOGL,TSLA" --analysis all
```

#### CLI Mode - Monte Carlo Only
```bash
python main_app.py --tickers "AAPL" --analysis montecarlo --simulations 2000 --days 365
```

#### CLI Mode - Export Results
```bash
python main_app.py --tickers "AAPL,MSFT,GOOGL,TSLA" --analysis all --export results.xlsx
```

#### CLI Mode - Custom Parameters
```bash
python main_app.py --tickers "AAPL,MSFT" --start "2021-01-01" --end "2023-12-31" --analysis all --simulations 1500
```

### Method 3: GUI Interface Features

When running in GUI mode, you can:

1. **Input Parameters**:
   - Enter stock tickers (comma-separated)
   - Set custom date ranges
   - Adjust Monte Carlo parameters (simulations, forecast days)

2. **Analysis Options**:
   - **Load Data**: Download and prepare stock data
   - **Calculate Metrics**: Risk and return analysis
   - **Show Correlation**: Correlation matrix and heatmap
   - **Monte Carlo (Single)**: Simulation for first stock
   - **Monte Carlo (All)**: Simulation for all stocks
   - **Efficient Frontier**: Portfolio optimization
   - **Run Full Analysis**: Complete workflow
   - **Export Results**: Save to Excel/CSV
   - **Clear Results**: Reset the display

3. **Real-time Status**: Status bar shows current operation progress

## Command Line Options

### Available Arguments
- `--tickers`: Comma-separated stock symbols (e.g., "AAPL,MSFT,GOOGL")
- `--start`: Start date in YYYY-MM-DD format (default: 2020-01-01)
- `--end`: End date in YYYY-MM-DD format (default: 2023-01-01)
- `--analysis`: Type of analysis - basic, portfolio, montecarlo, or all (default: all)
- `--export`: Export filename (.xlsx or .csv)
- `--simulations`: Number of Monte Carlo simulations (default: 1000)
- `--days`: Number of days to forecast (default: 252)
- `--all-stocks`: Run Monte Carlo for all stocks instead of just the first
- `--gui`: Force GUI mode even when other arguments are present

### Analysis Types
- **basic**: Risk and return metrics only
- **portfolio**: Portfolio analysis and optimization
- **montecarlo**: Monte Carlo simulation
- **all**: Complete analysis including all above

## Expected Output

### Console Output (CLI Mode)
- Data loading progress
- Risk and return metrics table
- Portfolio analysis results
- Monte Carlo simulation statistics
- Correlation matrix
- Efficient frontier results

### GUI Output
- Interactive results display
- Real-time status updates
- Tabbed interface for results and charts
- Export functionality

### Visual Output
Charts and graphs displayed for:
- Stock price performance
- Correlation heatmap
- Monte Carlo simulation paths
- Efficient frontier plot
- Risk analysis charts

### File Output
When using export options:
- Excel files with multiple sheets (Returns, Metrics, Correlation, Analysis Report)
- CSV files with returns data
- PNG files for saved charts

## Examples

### Example 1: Quick GUI Analysis
```bash
# Start GUI
python main_app.py

# In GUI:
# 1. Enter tickers: AAPL,MSFT,GOOGL
# 2. Click "Load Data"
# 3. Click "Run Full Analysis"
# 4. Click "Export Results"
```

### Example 2: Command Line Workflow
```bash
# Basic analysis
python main_app.py --tickers "AAPL,MSFT" --analysis basic

# Full analysis with export
python main_app.py --tickers "AAPL,MSFT,GOOGL,TSLA" --analysis all --export portfolio_analysis.xlsx

# Monte Carlo focus
python main_app.py --tickers "AAPL,TSLA" --analysis montecarlo --all-stocks --simulations 2000
```

### Example 3: Mixed Mode Usage
```bash
# Run CLI analysis first
python main_app.py --tickers "AAPL,MSFT" --analysis basic

# Then switch to GUI for detailed exploration
python main_app.py --gui
```

## Configuration

Edit `config.py` to customize default settings:
```python
# Default tickers
DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']

# Date range
START_DATE = '2020-01-01'
END_DATE = '2023-01-01'

# Risk-free rate
RISK_FREE_RATE = 0.03

# Monte Carlo parameters
MC_SIMULATIONS = 1000
MC_DAYS = 252
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors