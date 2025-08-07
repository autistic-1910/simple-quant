#!/usr/bin/env python3
"""
Add missing methods to the QuantitativeAnalysis class
"""

# Missing methods to add to quant_analysis.py and quant_analysis_fixed.py

missing_methods_code = '''
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
'''

def add_methods_to_file(filename):
    """Add missing methods to a file"""
    try:
        # Read the current file
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if methods already exist
        if 'def plot_price_performance(' in content:
            print(f"Methods already exist in {filename}")
            return True
        
        # Add methods before the last line (if it's just whitespace) or at the end
        lines = content.split('\n')
        
        # Find the last non-empty line
        insert_index = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip():
                insert_index = i + 1
                break
        
        # Insert the new methods
        method_lines = missing_methods_code.split('\n')
        lines[insert_index:insert_index] = method_lines
        
        # Write back to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"✓ Successfully added missing methods to {filename}")
        return True
        
    except Exception as e:
        print(f"Error adding methods to {filename}: {e}")
        return False

def main():
    """Add missing methods to both quant_analysis files"""
    print("Adding missing methods to QuantitativeAnalysis class...")
    
    files_to_update = [
        'quant_analysis.py',
        'quant_analysis_fixed.py'
    ]
    
    success_count = 0
    for filename in files_to_update:
        if add_methods_to_file(filename):
            success_count += 1
    
    print(f"\n✓ Successfully updated {success_count}/{len(files_to_update)} files")
    
    if success_count == len(files_to_update):
        print("\n🎉 All missing methods have been added!")
        print("\nThe following methods were added:")
        print("  - plot_price_performance()")
        print("  - efficient_frontier()")
        print("  - plot_monte_carlo()")
        print("  - monte_carlo_all_stocks()")
        print("\nYour application should now work without AttributeError exceptions!")
    else:
        print("\n⚠️  Some files could not be updated. Please check the errors above.")

if __name__ == "__main__":
    main()