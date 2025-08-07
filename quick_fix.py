#!/usr/bin/env python3
"""
Quick fix script for the date range issue
Run this to test with correct dates
"""

from quant_analysis import QuantitativeAnalysis
from datetime import datetime, timedelta

def quick_test():
    """Quick test with proper date range"""
    
    print("=== Quick Fix Test ===\n")
    
    # Calculate proper date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    
    print(f"Using corrected date range: {start_date} to {end_date}")
    
    # Test with a smaller set of tickers first
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    # Initialize analysis
    qa = QuantitativeAnalysis()
    
    # Load data with corrected dates
    print(f"\nTesting with tickers: {test_tickers}")
    data = qa.load_stock_data(test_tickers, start_date, end_date)
    
    if data is not None:
        print("✓ Data loaded successfully!")
        
        # Calculate returns
        returns = qa.calculate_returns()
        if returns is not None:
            print("✓ Returns calculated successfully!")
            
            # Calculate basic metrics
            metrics = qa.calculate_risk_return_metrics()
            if metrics is not None:
                print("✓ Risk metrics calculated successfully!")
                print("\nSample metrics:")
                print(metrics.round(4))
                
                return True
    
    print("✗ Test failed!")
    return False

def fix_all_tickers():
    """Test with all your original tickers using correct dates"""
    
    print("\n=== Testing All Tickers ===\n")
    
    # Your original tickers
    all_tickers = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'TSLA', 'JPM', 'UNH', 'XOM', 'BRK.B']
    
    # Calculate proper date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    
    print(f"Using corrected date range: {start_date} to {end_date}")
    print(f"Testing with all tickers: {all_tickers}")
    
    # Initialize analysis
    qa = QuantitativeAnalysis()
    
    # Load data with corrected dates
    data = qa.load_stock_data(all_tickers, start_date, end_date)
    
    if data is not None:
        print("✓ All data loaded successfully!")
        
        # Calculate returns
        returns = qa.calculate_returns()
        if returns is not None:
            print("✓ Returns calculated successfully!")
            
            # Calculate basic metrics
            metrics = qa.calculate_risk_return_metrics()
            if metrics is not None:
                print("✓ Risk metrics calculated successfully!")
                print(f"\nLoaded data for {len(metrics)} stocks:")
                for ticker in metrics.index:
                    annual_return = metrics.loc[ticker, 'Annual Return']
                    volatility = metrics.loc[ticker, 'Annual Volatility']
                    sharpe = metrics.loc[ticker, 'Sharpe Ratio']
                    print(f"  {ticker}: Return={annual_return:.2%}, Vol={volatility:.2%}, Sharpe={sharpe:.2f}")
                
                return True
    
    print("✗ Test with all tickers failed!")
    return False

if __name__ == "__main__":
    print("Quick Fix Script for Date Range Issue")
    print("=====================================")
    
    # Test with small set first
    if quick_test():
        # If successful, test with all tickers
        fix_all_tickers()
    
    print("\n=== Summary ===")
    print("The issue was caused by using future dates (2025-01-01).")
    print("Stock market data is only available up to the current date.")
    print("\nTo fix this permanently:")
    print("1. Use dates up to today's date")
    print("2. The config.py file has been updated with automatic date calculation")
    print("3. Always ensure end_date <= today's date")
    
    print(f"\nRecommended date range for analysis:")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    print(f"Start: {start_date}")
    print(f"End: {end_date}")