import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class QuantitativeAnalysis:
    def __init__(self):
        self.data = None
        self.returns = None
        self.portfolio_weights = None
        self.risk_free_rate = 0.03
        
    def load_stock_data(self, tickers, start_date='2020-01-01', end_date='2023-01-01'):
        """Load stock data for given tickers"""
        print(f"Loading data for: {tickers}")
        try:
            self.data = yf.download(tickers, start=start_date, end=end_date)
            if len(tickers) == 1:
                self.data = pd.DataFrame(self.data)
            print("Data loaded successfully!")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def calculate_returns(self, price_data=None):
        """Calculate returns from price data"""
        if price_data is None:
            price_data = self.data['Close'] if 'Close' in self.data.columns else self.data
        
        self.returns = price_data.pct_change().dropna()
        return self.returns
    
    def calculate_risk_return_metrics(self, returns_data=None):
        """Calculate annualized risk and return metrics"""
        if returns_data is None:
            returns_data = self.returns
            
        metrics = {}
        
        for column in returns_data.columns:
            daily_returns = returns_data[column].dropna()
            
            # Annualized return
            mean_return = daily_returns.mean() * 252
            
            # Annualized volatility
            volatility = daily_returns.std() * np.sqrt(252)
            
            # Sharpe ratio
            sharpe_ratio = (mean_return - self.risk_free_rate) / volatility
            
            # Value at Risk (95%)
            var_95 = np.percentile(daily_returns, 5)
            
            # Maximum drawdown
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
        
        return pd.DataFrame(metrics).T
    
    def portfolio_analysis(self, weights=None):
        """Perform portfolio analysis"""
        if weights is None:
            # Equal weights if not specified
            weights = np.array([1/len(self.returns.columns)] * len(self.returns.columns))
        
        self.portfolio_weights = weights
        
        # Calculate portfolio returns
        portfolio_returns = (self.returns * weights).sum(axis=1)
        
        # Covariance matrix (annualized)
        cov_matrix = self.returns.cov() * 252
        
        # Portfolio metrics
        portfolio_return = np.dot(weights, self.returns.mean() * 252)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        portfolio_sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
        
        return {
            'Portfolio Return': portfolio_return,
            'Portfolio Volatility': portfolio_vol,
            'Portfolio Sharpe Ratio': portfolio_sharpe,
            'Portfolio Returns': portfolio_returns
        }
    
    def monte_carlo_simulation(self, days=252, simulations=1000, ticker=None):
        """Run Monte Carlo simulation for price forecasting"""
        if ticker is None:
            ticker = self.returns.columns[0]
        
        daily_returns = self.returns[ticker].dropna()
        mean_return = daily_returns.mean()
        volatility = daily_returns.std()
        
        print(f"Running Monte Carlo simulation for {ticker}:")
        print(f"- Number of simulations: {simulations}")
        print(f"- Forecast period: {days} days")
        print(f"- Daily mean return: {mean_return:.6f}")
        print(f"- Daily volatility: {volatility:.6f}")
        
        # Generate random returns
        random_returns = np.random.normal(mean_return, volatility, (days, simulations))
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + random_returns, axis=0)
        
        # Print simulation statistics
        final_values = cumulative_returns[-1, :]
        print(f"\nSimulation Results Summary:")
        print(f"- Mean final value: {np.mean(final_values):.4f}")
        print(f"- Median final value: {np.median(final_values):.4f}")
        print(f"- 5th percentile: {np.percentile(final_values, 5):.4f}")
        print(f"- 95th percentile: {np.percentile(final_values, 95):.4f}")
        print(f"- Standard deviation: {np.std(final_values):.4f}")
        
        return cumulative_returns
    
    def monte_carlo_all_stocks(self, days=252, simulations=1000):
        """Run Monte Carlo simulation for all stocks"""
        results = {}
        
        for ticker in self.returns.columns:
            print(f"\n{'='*50}")
            print(f"Monte Carlo Simulation for {ticker}")
            print(f"{'='*50}")
            
            mc_result = self.monte_carlo_simulation(days, simulations, ticker)
            results[ticker] = mc_result
            
            # Plot individual results
            self.plot_monte_carlo(mc_result, ticker)
        
        return results
    
    def correlation_analysis(self):
        """Create correlation heatmap"""
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.returns.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title("Stock Returns Correlation Heatmap")
        plt.tight_layout()
        plt.show()
        return correlation_matrix
    
    def plot_price_performance(self):
        """Plot normalized price performance"""
        if 'Close' in self.data.columns:
            prices = self.data['Close']
        else:
            prices = self.data
            
        normalized_prices = prices / prices.iloc[0] * 100
        
        plt.figure(figsize=(12, 6))
        for column in normalized_prices.columns:
            plt.plot(normalized_prices.index, normalized_prices[column], label=column)
        
        plt.title("Normalized Price Performance (Base = 100)")
        plt.xlabel("Date")
        plt.ylabel("Normalized Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_monte_carlo(self, simulations_data, ticker, show_all_paths=False):
        """Plot Monte Carlo simulation results with enhanced visualization"""
        plt.figure(figsize=(15, 10))
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Simulation paths
        if show_all_paths:
            # Show all paths (can be overwhelming)
            for i in range(simulations_data.shape[1]):
                ax1.plot(simulations_data[:, i], alpha=0.05, color='blue', linewidth=0.5)
        else:
            # Show sample of paths
            sample_size = min(200, simulations_data.shape[1])
            sample_indices = np.random.choice(simulations_data.shape[1], sample_size, replace=False)
            for i in sample_indices:
                ax1.plot(simulations_data[:, i], alpha=0.1, color='blue', linewidth=0.5)
        
        # Plot percentiles
        percentiles = np.percentile(simulations_data, [5, 25, 50, 75, 95], axis=1)
        ax1.plot(percentiles[2], color='red', linewidth=3, label='Median (50th)')
        ax1.plot(percentiles[0], color='orange', linewidth=2, label='5th Percentile')
        ax1.plot(percentiles[4], color='green', linewidth=2, label='95th Percentile')
        ax1.fill_between(range(len(percentiles[0])), percentiles[0], percentiles[4], 
                        alpha=0.2, color='gray', label='90% Confidence Interval')
        
        ax1.set_title(f'Monte Carlo Paths - {ticker}')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Cumulative Return Factor')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Final value distribution
        final_values = simulations_data[-1, :]
        ax2.hist(final_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(final_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(final_values):.3f}')
        ax2.axvline(np.median(final_values), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(final_values):.3f}')
        ax2.set_title(f'Distribution of Final Values - {ticker}')
        ax2.set_xlabel('Final Value')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Percentile evolution
        percentiles_over_time = np.percentile(simulations_data, [10, 25, 50, 75, 90], axis=1)
        for i, p in enumerate([10, 25, 50, 75, 90]):
            ax3.plot(percentiles_over_time[i], label=f'{p}th Percentile')
        ax3.set_title(f'Percentile Evolution - {ticker}')
        ax3.set_xlabel('Days')
        ax3.set_ylabel('Cumulative Return Factor')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Risk metrics over time
        var_5 = np.percentile(simulations_data, 5, axis=1)
        var_1 = np.percentile(simulations_data, 1, axis=1)
        ax4.plot(var_5, color='orange', linewidth=2, label='VaR 5%')
        ax4.plot(var_1, color='red', linewidth=2, label='VaR 1%')
        ax4.fill_between(range(len(var_1)), var_1, 1, alpha=0.3, color='red', label='Extreme Loss Zone')
        ax4.set_title(f'Value at Risk Evolution - {ticker}')
        ax4.set_xlabel('Days')
        ax4.set_ylabel('Cumulative Return Factor')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed statistics
        print(f"\nDetailed Monte Carlo Statistics for {ticker}:")
        print(f"Total simulations: {simulations_data.shape[1]}")
        print(f"Forecast period: {simulations_data.shape[0]} days")
        print(f"\nFinal Value Statistics:")
        print(f"Mean: {np.mean(final_values):.4f}")
        print(f"Median: {np.median(final_values):.4f}")
        print(f"Standard Deviation: {np.std(final_values):.4f}")
        print(f"Minimum: {np.min(final_values):.4f}")
        print(f"Maximum: {np.max(final_values):.4f}")
        print(f"\nPercentiles:")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            print(f"{p}th percentile: {np.percentile(final_values, p):.4f}")
        
        # Probability of loss
        prob_loss = np.mean(final_values < 1) * 100
        print(f"\nProbability of loss: {prob_loss:.2f}%")
        
        # Expected shortfall (Conditional VaR)
        var_5_value = np.percentile(final_values, 5)
        expected_shortfall = np.mean(final_values[final_values <= var_5_value])
        print(f"Expected Shortfall (5%): {expected_shortfall:.4f}")
    
    def efficient_frontier(self, num_portfolios=10000):
        """Calculate and plot efficient frontier"""
        if len(self.returns.columns) < 2:
            print("Need at least 2 assets for efficient frontier analysis")
            return None
        
        num_assets = len(self.returns.columns)
        results = np.zeros((3, num_portfolios))
        
        # Calculate expected returns and covariance matrix
        expected_returns = self.returns.mean() * 252
        cov_matrix = self.returns.cov() * 252
        
        np.random.seed(42)
        for i in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            
            # Calculate portfolio return and volatility
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            portfolio_sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            
            results[0, i] = portfolio_return
            results[1, i] = portfolio_vol
            results[2, i] = portfolio_sharpe
        
        # Plot efficient frontier
        plt.figure(figsize=(10, 6))
        plt.scatter(results[1], results[0], c=results[2], cmap='viridis', alpha=0.6)
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility')
        plt.ylabel('Expected Return')
        plt.title('Efficient Frontier')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Find optimal portfolio (max Sharpe ratio)
        max_sharpe_idx = np.argmax(results[2])
        optimal_return = results[0, max_sharpe_idx]
        optimal_vol = results[1, max_sharpe_idx]
        optimal_sharpe = results[2, max_sharpe_idx]
        
        return {
            'Optimal Return': optimal_return,
            'Optimal Volatility': optimal_vol,
            'Optimal Sharpe Ratio': optimal_sharpe
        }

def main():
    """Main function to run the quantitative analysis"""
    # Initialize the analysis class
    quant = QuantitativeAnalysis()
    
    # Define tickers to analyze
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    
    print("=== Quantitative Finance Analysis ===\n")
    
    # 1. Load data
    data = quant.load_stock_data(tickers, start_date='2020-01-01', end_date='2023-01-01')
    if data is None:
        return
    
    # 2. Calculate returns
    returns = quant.calculate_returns()
    print(f"\nReturns calculated for {len(returns.columns)} assets")
    
    # 3. Risk and return metrics
    print("\n=== Risk and Return Metrics ===")
    metrics = quant.calculate_risk_return_metrics()
    print(metrics.round(4))
    
    # 4. Portfolio analysis
    print("\n=== Portfolio Analysis ===")
    portfolio_metrics = quant.portfolio_analysis()
    for key, value in portfolio_metrics.items():
        if key != 'Portfolio Returns':
            print(f"{key}: {value:.4f}")
    
    # 5. Correlation analysis
    print("\n=== Correlation Analysis ===")
    correlation_matrix = quant.correlation_analysis()
    
    # 6. Price performance plot
    print("\n=== Price Performance ===")
    quant.plot_price_performance()
    
    # 7. Monte Carlo simulation for ALL stocks
    print("\n=== Monte Carlo Simulation for All Stocks ===")
    mc_results_all = quant.monte_carlo_all_stocks(days=252, simulations=1000)
    
    # 8. Efficient frontier
    print("\n=== Efficient Frontier Analysis ===")
    efficient_frontier_results = quant.efficient_frontier()
    if efficient_frontier_results:
        print("Optimal Portfolio Metrics:")
        for key, value in efficient_frontier_results.items():
            print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()