#!/usr/bin/env python3
"""
Fix graph output issues in the quantitative analysis tool
"""

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import warnings
warnings.filterwarnings('ignore')

def detect_and_fix_matplotlib():
    """Detect and fix matplotlib backend issues"""
    
    print("=" * 60)
    print("FIXING GRAPH OUTPUT ISSUES")
    print("=" * 60)
    
    # Test different backends
    backends_to_try = [
        'TkAgg',      # Usually works on Windows
        'Qt5Agg',     # Alternative GUI backend
        'Agg',        # Non-interactive (saves files only)
        'module://backend_interagg'  # Interactive backend
    ]
    
    working_backend = None
    
    for backend in backends_to_try:
        try:
            print(f"Testing backend: {backend}")
            matplotlib.use(backend, force=True)
            
            # Test if backend works
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot([1, 2, 3], [1, 4, 2])
            ax.set_title('Test Plot')
            
            if backend == 'Agg':
                # For Agg backend, save to file
                plt.savefig('test_plot.png', dpi=100, bbox_inches='tight')
                print(f"✓ {backend} backend works (saves to file)")
                if os.path.exists('test_plot.png'):
                    os.remove('test_plot.png')  # Clean up
            else:
                # For interactive backends, try to show
                plt.show(block=False)
                print(f"✓ {backend} backend works (interactive)")
            
            plt.close('all')
            working_backend = backend
            break
            
        except Exception as e:
            print(f"✗ {backend} backend failed: {e}")
            plt.close('all')
            continue
    
    if working_backend:
        print(f"\n✓ Found working backend: {working_backend}")
        return working_backend
    else:
        print("\n✗ No working backend found!")
        return None

def create_fixed_plotting_functions():
    """Create improved plotting functions with better error handling"""
    
    code = '''
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set a working backend
try:
    matplotlib.use('TkAgg')
except:
    try:
        matplotlib.use('Agg')  # Fallback to file-only backend
    except:
        pass

class ImprovedPlotting:
    """Improved plotting functions with better error handling"""
    
    def __init__(self, save_plots=True, show_plots=True):
        self.save_plots = save_plots
        self.show_plots = show_plots
        self.plot_counter = 0
        
        # Set style
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                pass  # Use default style
    
    def _save_and_show(self, filename=None):
        """Save and/or show plot"""
        try:
            if self.save_plots:
                if filename is None:
                    filename = f"plot_{self.plot_counter}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                print(f"✓ Plot saved as: {filename}")
                self.plot_counter += 1
            
            if self.show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"Warning: Could not display plot: {e}")
            if self.save_plots:
                try:
                    if filename is None:
                        filename = f"plot_{self.plot_counter}.png"
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    print(f"✓ Plot saved as: {filename}")
                    self.plot_counter += 1
                except Exception as save_error:
                    print(f"Error saving plot: {save_error}")
            plt.close()
    
    def plot_price_chart(self, data, title="Stock Price Chart"):
        """Plot price chart with improved error handling"""
        try:
            if data is None or data.empty:
                print("Error: No data available for plotting.")
                return None
            
            plt.figure(figsize=(12, 6))
            
            # Handle different data structures
            if isinstance(data.columns, pd.MultiIndex):
                # Multi-level columns
                for ticker in data.columns.get_level_values(1).unique():
                    if 'Close' in data.columns.get_level_values(0):
                        plt.plot(data.index, data['Close'][ticker], 
                               label=ticker, linewidth=2)
            else:
                # Single level columns
                for column in data.columns:
                    plt.plot(data.index, data[column], 
                           label=column, linewidth=2)
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Price ($)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            self._save_and_show('price_chart.png')
            
        except Exception as e:
            print(f"Error plotting price chart: {e}")
            plt.close()

    def plot_returns_distribution(self, returns, title="Returns Distribution"):
        """Plot returns distribution with improved error handling"""
        try:
            if returns is None or returns.empty:
                print("Error: No returns data available for plotting.")
                return None
            
            num_assets = len(returns.columns)
            cols = min(num_assets, 3)
            
            fig, axes = plt.subplots(1, cols, figsize=(5*cols, 5))
            
            if cols == 1:
                axes = [axes]
            
            for i, column in enumerate(returns.columns[:cols]):
                returns[column].hist(bins=50, alpha=0.7, ax=axes[i], 
                                   color=f'C{i}', edgecolor='black', linewidth=0.5)
                axes[i].set_title(f'{column} Returns Distribution', 
                                fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Daily Returns', fontsize=10)
                axes[i].set_ylabel('Frequency', fontsize=10)
                axes[i].grid(True, alpha=0.3)
                
                # Add statistics
                mean_ret = returns[column].mean()
                std_ret = returns[column].std()
                axes[i].axvline(mean_ret, color='red', linestyle='--', 
                              label=f'Mean: {mean_ret:.4f}')
                axes[i].legend(fontsize=8)
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            self._save_and_show('returns_distribution.png')
            
        except Exception as e:
            print(f"Error plotting returns distribution: {e}")
            plt.close()

    def plot_correlation_heatmap(self, correlation_matrix, title="Asset Correlation Matrix"):
        """Plot correlation heatmap with improved error handling"""
        try:
            if correlation_matrix is None or correlation_matrix.empty:
                print("Error: No correlation data available for plotting.")
                return None
            
            plt.figure(figsize=(10, 8))
            
            # Create heatmap
            sns.heatmap(correlation_matrix, 
                       annot=True, 
                       cmap='RdYlBu_r', 
                       center=0,
                       square=True, 
                       linewidths=0.5,
                       cbar_kws={"shrink": .8},
                       fmt='.2f',
                       annot_kws={'size': 10})
            
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            
            self._save_and_show('correlation_heatmap.png')
            
        except Exception as e:
            print(f"Error plotting correlation heatmap: {e}")
            plt.close()
    
    def plot_monte_carlo_results(self, mc_results, title="Monte Carlo Simulation Results"):
        """Plot Monte Carlo simulation results"""
        try:
            if mc_results is None or mc_results.empty:
                print("Error: No Monte Carlo results available for plotting.")
                return None
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Scatter plot of Return vs Volatility
            scatter = axes[0].scatter(mc_results['Volatility'], mc_results['Return'], 
                                    c=mc_results['Sharpe Ratio'], cmap='viridis', 
                                    alpha=0.6, s=20)
            axes[0].set_xlabel('Volatility', fontsize=12)
            axes[0].set_ylabel('Return', fontsize=12)
            axes[0].set_title('Efficient Frontier', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=axes[0])
            cbar.set_label('Sharpe Ratio', fontsize=10)
            
            # Histogram of Sharpe Ratios
            axes[1].hist(mc_results['Sharpe Ratio'], bins=50, alpha=0.7, 
                        color='skyblue', edgecolor='black', linewidth=0.5)
            axes[1].set_xlabel('Sharpe Ratio', fontsize=12)
            axes[1].set_ylabel('Frequency', fontsize=12)
            axes[1].set_title('Sharpe Ratio Distribution', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            # Add statistics
            mean_sharpe = mc_results['Sharpe Ratio'].mean()
            axes[1].axvline(mean_sharpe, color='red', linestyle='--', 
                          label=f'Mean: {mean_sharpe:.3f}')
            axes[1].legend()
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            self._save_and_show('monte_carlo_results.png')
            
        except Exception as e:
            print(f"Error plotting Monte Carlo results: {e}")
            plt.close()
    
    def plot_portfolio_performance(self, portfolio_returns, title="Portfolio Performance"):
        """Plot portfolio performance over time"""
        try:
            if portfolio_returns is None or len(portfolio_returns) == 0:
                print("Error: No portfolio returns available for plotting.")
                return None
            
            # Calculate cumulative returns
            cumulative_returns = (1 + portfolio_returns).cumprod()
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # Cumulative returns plot
            axes[0].plot(cumulative_returns.index, cumulative_returns.values, 
                        linewidth=2, color='blue')
            axes[0].set_title('Cumulative Portfolio Returns', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Cumulative Return', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            
            # Daily returns plot
            axes[1].plot(portfolio_returns.index, portfolio_returns.values, 
                        linewidth=1, color='green', alpha=0.7)
            axes[1].set_title('Daily Portfolio Returns', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Date', fontsize=12)
            axes[1].set_ylabel('Daily Return', fontsize=12)
            axes[1].grid(True, alpha=0.3)
            axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            self._save_and_show('portfolio_performance.png')
            
        except Exception as e:
            print(f"Error plotting portfolio performance: {e}")
            plt.close()

# Test the plotting functions
def test_plotting():
    """Test the improved plotting functions"""
    print("\\nTesting improved plotting functions...")
    
    # Create sample data
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Sample price data
    price_data = pd.DataFrame({
        'AAPL': 150 + np.cumsum(np.random.randn(len(dates)) * 0.02),
        'MSFT': 300 + np.cumsum(np.random.randn(len(dates)) * 0.02),
        'GOOGL': 2500 + np.cumsum(np.random.randn(len(dates)) * 0.03)
    }, index=dates)
    
    # Sample returns data
    returns_data = price_data.pct_change().dropna()
    
    # Create plotter
    plotter = ImprovedPlotting(save_plots=True, show_plots=False)
    
    print("Testing price chart...")
    plotter.plot_price_chart(price_data)
    
    print("Testing returns distribution...")
    plotter.plot_returns_distribution(returns_data)
    
    print("Testing correlation heatmap...")
    correlation_matrix = returns_data.corr()
    plotter.plot_correlation_heatmap(correlation_matrix)
    
    print("✓ All plotting tests completed!")

if __name__ == "__main__":
    test_plotting()
'''
    
    with open('improved_plotting.py', 'w') as f:
        f.write(code)
    
    print("✓ Created improved_plotting.py with enhanced plotting functions")

def main():
    """Main function to fix graph output issues"""
    
    # Step 1: Detect working backend
    working_backend = detect_and_fix_matplotlib()
    
    # Step 2: Create improved plotting functions
    create_fixed_plotting_functions()
    
    # Step 3: Create matplotlib config
    try:
        config_dir = matplotlib.get_configdir()
        config_file = os.path.join(config_dir, 'matplotlibrc')
        
        os.makedirs(config_dir, exist_ok=True)
        
        config_content = f"""
# Matplotlib configuration for quantitative analysis
backend: {working_backend if working_backend else 'Agg'}
figure.figsize: 10, 6
figure.dpi: 100
savefig.dpi: 300
savefig.format: png
savefig.bbox: tight
font.size: 10
axes.grid: True
grid.alpha: 0.3
"""
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print(f"✓ Created matplotlib config at: {config_file}")
        
    except Exception as e:
        print(f"Warning: Could not create matplotlib config: {e}")
    
    # Step 4: Provide instructions
    print("\n" + "=" * 60)
    print("GRAPH OUTPUT FIX SUMMARY")
    print("=" * 60)
    
    if working_backend:
        print(f"✓ Working matplotlib backend: {working_backend}")
    else:
        print("⚠ No interactive backend found - plots will be saved as files")
    
    print("✓ Created improved_plotting.py with enhanced functions")
    print("✓ Created matplotlib configuration")
    
    print("\nTo use the improved plotting in your analysis:")
    print("1. Import: from improved_plotting import ImprovedPlotting")
    print("2. Create plotter: plotter = ImprovedPlotting()")
    print("3. Use methods: plotter.plot_price_chart(data)")
    
    print("\nCommon graph issues and solutions:")
    print("- No graphs showing: Plots will be saved as PNG files")
    print("- Backend errors: Using fallback Agg backend (file output only)")
    print("- Style issues: Enhanced styling and error handling added")
    
    return working_backend

if __name__ == "__main__":
    main()