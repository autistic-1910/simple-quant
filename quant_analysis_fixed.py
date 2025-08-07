import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
from datetime import datetime, timedelta
import time
warnings.filterwarnings('ignore')

class QuantitativeAnalysis:
    def __init__(self):
        self.data = None
        self.returns = None
        self.portfolio_weights = None
        self.risk_free_rate = 0.03
        
    def validate_tickers(self, tickers):
        """Validate ticker symbols"""
        valid_tickers = []
        invalid_tickers = []
        
        for ticker in tickers:
            ticker = ticker.strip().upper()
            if ticker and len(ticker) <= 10:  # Basic validation
                valid_tickers.append(ticker)
            else:
                invalid_tickers.append(ticker)
        
        if invalid_tickers:
            print(f"Warning: Invalid tickers removed: {invalid_tickers}")
        
        return valid_tickers
    
    def validate_dates(self, start_date, end_date):
        """Validate and adjust date ranges"""
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            today = pd.Timestamp.now()
            
            # Ensure start date is not in the future
            if start > today:
                start = today - timedelta(days=365*2)  # Default to 2 years ago
                print(f"Warning: Start date adjusted to {start.strftime('%Y-%m-%d')}")
            
            # Ensure end date is not in the future
            if end > today:
                end = today
                print(f"Warning: End date adjusted to {end.strftime('%Y-%m-%d')}")
            
            # Ensure start is before end
            if start >= end:
                end = start + timedelta(days=365)
                if end > today:
                    end = today
                    start = end - timedelta(days=365)
                print(f"Warning: Date range adjusted to {start.strftime('%Y-%m-%d')} - {end.strftime('%Y-%m-%d')}")
            
            return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
            
        except Exception as e:
            print(f"Error validating dates: {e}")
            # Return default dates
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
            return start_date, end_date
        
    def load_stock_data(self, tickers, start_date='2020-01-01', end_date='2023-01-01', max_retries=3):
        """Load stock data with FIXED data combination logic"""
        
        # Validate inputs
        if isinstance(tickers, str):
            tickers = [tickers]
        
        valid_tickers = self.validate_tickers(tickers)
        if not valid_tickers:
            print("Error: No valid tickers provided!")
            return None
        
        start_date, end_date = self.validate_dates(start_date, end_date)
        
        print(f"Loading data for: {valid_tickers}")
        print(f"Date range: {start_date} to {end_date}")
        
        # Try downloading all tickers at once first (more efficient)
        try:
            print("Attempting bulk download...")
            bulk_data = yf.download(valid_tickers, start=start_date, end=end_date, progress=False, group_by='ticker')
            
            if bulk_data is not None and not bulk_data.empty:
                # Handle single vs multiple tickers
                if len(valid_tickers) == 1:
                    # Single ticker - data structure is simpler
                    self.data = bulk_data
                    if len(bulk_data) >= 10:
                        print(f"✓ Successfully loaded {valid_tickers[0]} ({len(bulk_data)} data points)")
                        print(f"✓ Data loaded successfully! Shape: {self.data.shape}")
                        print(f"Date range: {self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}")
                        return self.data
                else:
                    # Multiple tickers - extract Close prices properly
                    print("Processing multiple tickers...")
                    close_data = {}
                    successful_tickers = []
                    
                    for ticker in valid_tickers:
                        try:
                            if ticker in bulk_data.columns.get_level_values(0):
                                ticker_data = bulk_data[ticker]
                                if 'Close' in ticker_data.columns and not ticker_data['Close'].isna().all():
                                    close_data[ticker] = ticker_data['Close']
                                    successful_tickers.append(ticker)
                                    print(f"✓ Successfully processed {ticker}")
                                else:
                                    print(f"✗ {ticker}: No valid Close data")
                            else:
                                print(f"✗ {ticker}: Not found in bulk data")
                        except Exception as e:
                            print(f"✗ {ticker}: Error processing - {str(e)}")
                    
                    if close_data:
                        # Create DataFrame from close prices
                        self.data = pd.DataFrame(close_data)
                        
                        # Remove rows with any NaN values
                        initial_length = len(self.data)
                        self.data = self.data.dropna()
                        final_length = len(self.data)
                        
                        if final_length < initial_length:
                            print(f"Removed {initial_length - final_length} rows with missing data")
                        
                        if final_length >= 10:
                            print(f"✓ Successfully loaded: {successful_tickers}")
                            print(f"✓ Data loaded successfully! Shape: {self.data.shape}")
                            print(f"Date range: {self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}")
                            return self.data
                        else:
                            print("Error: Insufficient data after cleaning!")
                    else:
                        print("Error: No valid close price data found!")
            
        except Exception as e:
            print(f"Bulk download failed: {e}")
        
        # Fallback: Individual ticker loading
        print("Falling back to individual ticker loading...")
        successful_tickers = []
        all_close_data = {}
        
        for ticker in valid_tickers:
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    print(f"Attempting to load {ticker} (attempt {retry_count + 1}/{max_retries})")
                    
                    # Download data for individual ticker
                    ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    
                    # Validate the downloaded data
                    if ticker_data is not None and not ticker_data.empty and len(ticker_data) >= 10:
                        if 'Close' in ticker_data.columns:
                            close_prices = ticker_data['Close'].dropna()
                            if len(close_prices) >= 10:
                                all_close_data[ticker] = close_prices
                                successful_tickers.append(ticker)
                                success = True
                                print(f"✓ Successfully loaded {ticker} ({len(close_prices)} data points)")
                            else:
                                print(f"✗ {ticker}: Insufficient valid close prices")
                        else:
                            print(f"✗ {ticker}: No Close column found")
                    else:
                        print(f"✗ {ticker}: No data or insufficient data points")
                        
                except Exception as e:
                    print(f"✗ {ticker}: Error loading data - {str(e)}")
                
                if not success:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"Retrying {ticker} in 2 seconds...")
                        time.sleep(2)
        
        # Combine individual ticker data
        if all_close_data:
            try:
                # Create DataFrame from close prices with proper alignment
                self.data = pd.DataFrame(all_close_data)
                
                # Remove rows with any NaN values
                initial_length = len(self.data)
                self.data = self.data.dropna()
                final_length = len(self.data)
                
                if final_length < initial_length:
                    print(f"Removed {initial_length - final_length} rows with missing data")
                
                if final_length >= 10:
                    print(f"✓ Successfully loaded: {successful_tickers}")
                    print(f"✓ Data loaded successfully! Shape: {self.data.shape}")
                    print(f"Date range: {self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}")
                    return self.data
                else:
                    print("Error: Insufficient data after cleaning!")
                    return None
            except Exception as e:
                print(f"Error combining individual ticker data: {e}")
                return None
        else:
            print("Error: No data could be loaded for any ticker!")
            return None
    
    def calculate_returns(self, price_data=None):
        """Calculate returns with enhanced validation"""
        try:
            if price_data is None:
                if self.data is None:
                    print("Error: No data available. Please load data first.")
                    return None
                price_data = self.data
            
            # Calculate returns directly from price data
            returns = price_data.pct_change()
            
            # Remove infinite and NaN values
            returns = returns.replace([np.inf, -np.inf], np.nan)
            returns = returns.dropna()
            
            if returns.empty:
                print("Error: No valid returns could be calculated!")
                return None
            
            # Check for excessive volatility (potential data errors)
            for col in returns.columns:
                col_returns = returns[col]
                if col_returns.std() > 1.0:  # Daily returns with >100% volatility
                    print(f"Warning: {col} shows extremely high volatility. Data may be incorrect.")
            
            self.returns = returns
            print(f"✓ Returns calculated successfully! Shape: {returns.shape}")
            return returns
            
        except Exception as e:
            print(f"Error calculating returns: {e}")
            return None
    
    def calculate_risk_return_metrics(self, returns_data=None):
        """Calculate risk and return metrics with enhanced error handling"""
        try:
            if returns_data is None:
                if self.returns is None:
                    print("Error: No returns data available. Please calculate returns first.")
                    return None
                returns_data = self.returns
            
            if returns_data.empty:
                print("Error: Returns data is empty!")
                return None
            
            metrics = {}
            
            for column in returns_data.columns:
                try:
                    daily_returns = returns_data[column].dropna()
                    
                    if len(daily_returns) < 10:
                        print(f"Warning: Insufficient data for {column} ({len(daily_returns)} points)")
                        continue
                    
                    # Calculate metrics with error handling
                    mean_return = daily_returns.mean() * 252
                    volatility = daily_returns.std() * np.sqrt(252)
                    
                    # Handle division by zero
                    if volatility == 0:
                        sharpe_ratio = 0
                    else:
                        sharpe_ratio = (mean_return - self.risk_free_rate) / volatility
                    
                    # VaR calculation
                    if len(daily_returns) >= 20:
                        var_95 = np.percentile(daily_returns, 5)
                    else:
                        var_95 = daily_returns.min()
                    
                    # Maximum drawdown calculation
                    cumulative_returns = (1 + daily_returns).cumprod()
                    rolling_max = cumulative_returns.expanding().max()
                    drawdown = (cumulative_returns - rolling_max) / rolling_max
                    max_drawdown = drawdown.min()
                    
                    metrics[column] = {
                        'Annual Return': mean_return,
                        'Annual Volatility': volatility,
                        'Sharpe Ratio': sharpe_ratio,
                        'VaR (95%)': var_95,
                        'Max Drawdown': max_drawdown
                    }
                    
                except Exception as e:
                    print(f"Error calculating metrics for {column}: {e}")
                    continue
            
            if not metrics:
                print("Error: No metrics could be calculated!")
                return None
            
            result = pd.DataFrame(metrics).T
            print(f"✓ Risk/return metrics calculated for {len(metrics)} assets")
            return result
            
        except Exception as e:
            print(f"Error in risk/return calculation: {e}")
            return None

    def portfolio_analysis(self, weights=None):
        """Perform portfolio analysis with enhanced validation"""
        try:
            if self.returns is None or self.returns.empty:
                print("Error: No returns data available for portfolio analysis.")
                return None
            
            if weights is None:
                # Equal weights if not specified
                weights = np.array([1/len(self.returns.columns)] * len(self.returns.columns))
            
            if len(weights) != len(self.returns.columns):
                print(f"Error: Number of weights ({len(weights)}) doesn't match number of assets ({len(self.returns.columns)})")
                return None
            
            self.portfolio_weights = weights
            
            # Calculate portfolio returns
            portfolio_returns = (self.returns * weights).sum(axis=1)
            
            # Portfolio metrics
            portfolio_mean = portfolio_returns.mean() * 252
            portfolio_vol = portfolio_returns.std() * np.sqrt(252)
            portfolio_sharpe = (portfolio_mean - self.risk_free_rate) / portfolio_vol if portfolio_vol != 0 else 0
            
            # Portfolio VaR
            portfolio_var = np.percentile(portfolio_returns, 5)
            
            # Portfolio max drawdown
            cumulative_portfolio = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_portfolio.expanding().max()
            drawdown = (cumulative_portfolio - rolling_max) / rolling_max
            portfolio_max_dd = drawdown.min()
            
            portfolio_metrics = {
                'Portfolio Return': portfolio_mean,
                'Portfolio Volatility': portfolio_vol,
                'Portfolio Sharpe Ratio': portfolio_sharpe,
                'Portfolio VaR (95%)': portfolio_var,
                'Portfolio Max Drawdown': portfolio_max_dd
            }
            
            print("✓ Portfolio analysis completed successfully")
            return portfolio_metrics, portfolio_returns
            
        except Exception as e:
            print(f"Error in portfolio analysis: {e}")
            return None, None

    def monte_carlo_simulation(self, num_simulations=1000, time_horizon=252):
        """Monte Carlo simulation with enhanced error handling"""
        try:
            if self.returns is None or self.returns.empty:
                print("Error: No returns data available for Monte Carlo simulation.")
                return None
            
            # Calculate mean returns and covariance matrix
            mean_returns = self.returns.mean()
            cov_matrix = self.returns.cov()
            
            # Validate inputs
            if mean_returns.isna().any() or cov_matrix.isna().any().any():
                print("Error: NaN values in returns or covariance matrix")
                return None
            
            num_assets = len(mean_returns)
            results = np.zeros((num_simulations, num_assets))
            
            print(f"Running Monte Carlo simulation with {num_simulations} simulations...")
            
            for i in range(num_simulations):
                # Generate random weights
                weights = np.random.random(num_assets)
                weights /= np.sum(weights)
                
                # Calculate portfolio return and volatility
                portfolio_return = np.sum(weights * mean_returns) * 252
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
                
                # Store results
                results[i, 0] = portfolio_return
                results[i, 1] = portfolio_vol
                if portfolio_vol != 0:
                    results[i, 2] = (portfolio_return - self.risk_free_rate) / portfolio_vol
                else:
                    results[i, 2] = 0
            
            # Create results DataFrame
            mc_results = pd.DataFrame(results[:, :3], columns=['Return', 'Volatility', 'Sharpe Ratio'])
            
            print("✓ Monte Carlo simulation completed successfully")
            return mc_results
            
        except Exception as e:
            print(f"Error in Monte Carlo simulation: {e}")
            return None

    def correlation_analysis(self):
        """Correlation analysis with enhanced validation"""
        try:
            if self.returns is None or self.returns.empty:
                print("Error: No returns data available for correlation analysis.")
                return None
            
            correlation_matrix = self.returns.corr()
            
            if correlation_matrix.isna().any().any():
                print("Warning: NaN values in correlation matrix")
                correlation_matrix = correlation_matrix.fillna(0)
            
            print("✓ Correlation analysis completed successfully")
            return correlation_matrix
            
        except Exception as e:
            print(f"Error in correlation analysis: {e}")
            return None

    def plot_price_chart(self):
        """Plot price chart with error handling"""
        try:
            if self.data is None or self.data.empty:
                print("Error: No data available for plotting.")
                return None
            
            plt.figure(figsize=(12, 6))
            
            # Handle different data structures
            if isinstance(self.data.columns, pd.MultiIndex):
                # Multi-level columns
                for ticker in self.data.columns.get_level_values(1).unique():
                    plt.plot(self.data.index, self.data['Close'][ticker], label=ticker)
            else:
                # Single level columns
                for column in self.data.columns:
                    plt.plot(self.data.index, self.data[column], label=column)
            
            plt.title('Stock Price Chart')
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting price chart: {e}")

    def plot_returns_distribution(self):
        """Plot returns distribution with error handling"""
        try:
            if self.returns is None or self.returns.empty:
                print("Error: No returns data available for plotting.")
                return None
            
            num_assets = len(self.returns.columns)
            fig, axes = plt.subplots(1, min(num_assets, 3), figsize=(15, 5))
            
            if num_assets == 1:
                axes = [axes]
            
            for i, column in enumerate(self.returns.columns[:3]):  # Plot max 3 assets
                self.returns[column].hist(bins=50, alpha=0.7, ax=axes[i])
                axes[i].set_title(f'{column} Returns Distribution')
                axes[i].set_xlabel('Daily Returns')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting returns distribution: {e}")

    def plot_correlation_heatmap(self):
        """Plot correlation heatmap with error handling"""
        try:
            correlation_matrix = self.correlation_analysis()
            if correlation_matrix is None:
                return None
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5)
            plt.title('Asset Correlation Matrix')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting correlation heatmap: {e}")

    def plot_price_performance(self):
        """Plot price performance chart - alias for plot_price_chart"""
        try:
            return self.plot_price_chart()
        except Exception as e:
            print(f"Error in plot_price_performance: {e}")
            return None
    
    def efficient_frontier(self, num_portfolios=10000):
        """Calculate and plot efficient frontier"""
        try:
            if self.returns is None or self.returns.empty:
                print("Error: No returns data available for efficient frontier.")
                return None
            
            if len(self.returns.columns) < 2:
                print("Error: Need at least 2 assets for efficient frontier.")
                return None
            
            print(f"Calculating efficient frontier with {num_portfolios} portfolios...")
            
            # Calculate mean returns and covariance matrix
            mean_returns = self.returns.mean() * 252  # Annualized
            cov_matrix = self.returns.cov() * 252  # Annualized
            
            # Validate inputs
            if mean_returns.isna().any() or cov_matrix.isna().any().any():
                print("Error: NaN values in returns or covariance matrix")
                return None
            
            num_assets = len(mean_returns)
            results = np.zeros((num_portfolios, 3))
            weights_array = np.zeros((num_portfolios, num_assets))
            
            # Generate random portfolios
            for i in range(num_portfolios):
                # Generate random weights
                weights = np.random.random(num_assets)
                weights /= np.sum(weights)
                weights_array[i] = weights
                
                # Calculate portfolio return and volatility
                portfolio_return = np.sum(weights * mean_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                # Store results
                results[i, 0] = portfolio_return
                results[i, 1] = portfolio_vol
                if portfolio_vol != 0:
                    results[i, 2] = (portfolio_return - self.risk_free_rate) / portfolio_vol
                else:
                    results[i, 2] = 0
            
            # Find optimal portfolios
            max_sharpe_idx = np.argmax(results[:, 2])
            min_vol_idx = np.argmin(results[:, 1])
            
            # Create results
            efficient_frontier_results = {
                'Max Sharpe Return': results[max_sharpe_idx, 0],
                'Max Sharpe Volatility': results[max_sharpe_idx, 1],
                'Max Sharpe Ratio': results[max_sharpe_idx, 2],
                'Min Vol Return': results[min_vol_idx, 0],
                'Min Vol Volatility': results[min_vol_idx, 1],
                'Min Vol Sharpe': results[min_vol_idx, 2]
            }
            
            # Plot efficient frontier
            try:
                plt.figure(figsize=(12, 8))
                scatter = plt.scatter(results[:, 1], results[:, 0], c=results[:, 2], 
                                    cmap='viridis', alpha=0.6)
                plt.colorbar(scatter, label='Sharpe Ratio')
                
                # Highlight optimal portfolios
                plt.scatter(results[max_sharpe_idx, 1], results[max_sharpe_idx, 0], 
                           marker='*', color='red', s=500, label='Max Sharpe')
                plt.scatter(results[min_vol_idx, 1], results[min_vol_idx, 0], 
                           marker='*', color='blue', s=500, label='Min Volatility')
                
                plt.title('Efficient Frontier')
                plt.xlabel('Volatility (Risk)')
                plt.ylabel('Expected Return')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
            except Exception as plot_error:
                print(f"Warning: Could not plot efficient frontier: {plot_error}")
            
            print("✓ Efficient frontier analysis completed successfully")
            return efficient_frontier_results
            
        except Exception as e:
            print(f"Error in efficient frontier calculation: {e}")
            return None
    
    def plot_monte_carlo(self, mc_results, ticker=None):
        """Plot Monte Carlo simulation results"""
        try:
            if mc_results is None or mc_results.empty:
                print("Error: No Monte Carlo results to plot.")
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Return vs Volatility
            scatter = axes[0, 0].scatter(mc_results['Volatility'], mc_results['Return'], 
                                       c=mc_results['Sharpe Ratio'], cmap='viridis', alpha=0.6)
            axes[0, 0].set_xlabel('Volatility')
            axes[0, 0].set_ylabel('Return')
            axes[0, 0].set_title('Monte Carlo: Return vs Volatility')
            axes[0, 0].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[0, 0], label='Sharpe Ratio')
            
            # Plot 2: Return Distribution
            axes[0, 1].hist(mc_results['Return'], bins=50, alpha=0.7, color='blue')
            axes[0, 1].set_xlabel('Return')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Return Distribution')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Volatility Distribution
            axes[1, 0].hist(mc_results['Volatility'], bins=50, alpha=0.7, color='green')
            axes[1, 0].set_xlabel('Volatility')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Volatility Distribution')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Sharpe Ratio Distribution
            axes[1, 1].hist(mc_results['Sharpe Ratio'], bins=50, alpha=0.7, color='red')
            axes[1, 1].set_xlabel('Sharpe Ratio')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Sharpe Ratio Distribution')
            axes[1, 1].grid(True, alpha=0.3)
            
            title_suffix = f" - {ticker}" if ticker else ""
            fig.suptitle(f'Monte Carlo Simulation Results{title_suffix}', fontsize=16)
            plt.tight_layout()
            plt.show()
            
            print("✓ Monte Carlo plots displayed successfully")
            
        except Exception as e:
            print(f"Error plotting Monte Carlo results: {e}")
    
    def monte_carlo_all_stocks(self, days=252, simulations=1000):
        """Run Monte Carlo simulation for all stocks individually"""
        try:
            if self.returns is None or self.returns.empty:
                print("Error: No returns data available for Monte Carlo simulation.")
                return None
            
            all_results = {}
            
            for ticker in self.returns.columns:
                print(f"Running Monte Carlo for {ticker}...")
                
                try:
                    # Get individual stock returns
                    stock_returns = self.returns[ticker].dropna()
                    
                    if len(stock_returns) < 30:  # Need sufficient data
                        print(f"Warning: Insufficient data for {ticker}")
                        continue
                    
                    # Calculate statistics
                    mean_return = stock_returns.mean()
                    std_return = stock_returns.std()
                    
                    # Run simulation
                    simulation_results = []
                    
                    for _ in range(simulations):
                        # Generate random returns
                        random_returns = np.random.normal(mean_return, std_return, days)
                        
                        # Calculate cumulative return
                        cumulative_return = (1 + random_returns).prod() - 1
                        annualized_return = (1 + cumulative_return) ** (252/days) - 1
                        annualized_vol = std_return * np.sqrt(252)
                        sharpe = (annualized_return - self.risk_free_rate) / annualized_vol if annualized_vol != 0 else 0
                        
                        simulation_results.append({
                            'Return': annualized_return,
                            'Volatility': annualized_vol,
                            'Sharpe Ratio': sharpe
                        })
                    
                    # Store results
                    all_results[ticker] = pd.DataFrame(simulation_results)
                    print(f"✓ Monte Carlo completed for {ticker}")
                    
                except Exception as ticker_error:
                    print(f"Error with {ticker}: {ticker_error}")
                    continue
            
            if all_results:
                print(f"✓ Monte Carlo simulation completed for {len(all_results)} stocks")
                
                # Plot results for each stock
                for ticker, results in all_results.items():
                    self.plot_monte_carlo(results, ticker)
                
                return all_results
            else:
                print("Error: No successful Monte Carlo simulations")
                return None
                
        except Exception as e:
            print(f"Error in Monte Carlo all stocks: {e}")
            return None
