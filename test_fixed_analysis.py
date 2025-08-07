#!/usr/bin/env python3
"""
Test script for the fixed quantitative analysis
"""

import sys
import os
from datetime import datetime, timedelta

# Import the fixed analysis class
from quant_analysis_fixed import QuantitativeAnalysis

def test_fixed_analysis():
    """Test the fixed analysis with current dates"""
    
    print("=" * 60)
    print("TESTING FIXED QUANTITATIVE ANALYSIS")
    print("=" * 60)
    
    # Create analysis instance
    qa = QuantitativeAnalysis()
    
    # Use safe dates (current date and 1 year back)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f"Using date range: {start_date} to {end_date}")
    print()
    
    # Test with a few reliable tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    print("Step 1: Loading stock data...")
    data = qa.load_stock_data(test_tickers, start_date=start_date, end_date=end_date)
    
    if data is not None:
        print(f"✓ Data loaded successfully!")
        print(f"  Shape: {data.shape}")
        print(f"  Columns: {list(data.columns)}")
        print(f"  Date range: {data.index[0]} to {data.index[-1]}")
        print(f"  Sample data:")
        print(data.head())
        print()
        
        print("Step 2: Calculating returns...")
        returns = qa.calculate_returns()
        
        if returns is not None:
            print(f"✓ Returns calculated successfully!")
            print(f"  Shape: {returns.shape}")
            print(f"  Sample returns:")
            print(returns.head())
            print()
            
            print("Step 3: Calculating risk/return metrics...")
            metrics = qa.calculate_risk_return_metrics()
            
            if metrics is not None:
                print(f"✓ Risk/return metrics calculated successfully!")
                print(metrics)
                print()
                
                print("Step 4: Portfolio analysis...")
                portfolio_metrics, portfolio_returns = qa.portfolio_analysis()
                
                if portfolio_metrics is not None:
                    print(f"✓ Portfolio analysis completed successfully!")
                    for key, value in portfolio_metrics.items():
                        print(f"  {key}: {value:.4f}")
                    print()
                    
                    print("Step 5: Monte Carlo simulation...")
                    mc_results = qa.monte_carlo_simulation(num_simulations=100)
                    
                    if mc_results is not None:
                        print(f"✓ Monte Carlo simulation completed successfully!")
                        print(f"  Results shape: {mc_results.shape}")
                        print(f"  Sample results:")
                        print(mc_results.head())
                        print()
                        
                        print("=" * 60)
                        print("✓ ALL TESTS PASSED! The fix is working correctly.")
                        print("=" * 60)
                        return True
                    else:
                        print("✗ Monte Carlo simulation failed")
                else:
                    print("✗ Portfolio analysis failed")
            else:
                print("✗ Risk/return metrics calculation failed")
        else:
            print("✗ Returns calculation failed")
    else:
        print("✗ Data loading failed")
    
    print("=" * 60)
    print("✗ TESTS FAILED")
    print("=" * 60)
    return False

if __name__ == "__main__":
    test_fixed_analysis()