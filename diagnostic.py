import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

def diagnose_data_issues():
    """Diagnostic script to identify common data loading issues"""
    
    print("=== Quantitative Finance Tool Diagnostic ===\n")
    
    # Test 1: Check internet connectivity and yfinance
    print("1. Testing yfinance connectivity...")
    try:
        test_data = yf.download("AAPL", start="2023-01-01", end="2023-01-10", progress=False, auto_adjust=True)
        if test_data is not None and not test_data.empty:
            print("✓ yfinance is working correctly")
        else:
            print("✗ yfinance returned empty data")
            return False
    except Exception as e:
        print(f"✗ yfinance error: {e}")
        return False
    
    # Test 2: Check date handling
    print("\n2. Testing date validation...")
    try:
        today = datetime.now()
        start_date = (today - timedelta(days=365)).strftime('%Y-%m-%d')
        end_date = today.strftime('%Y-%m-%d')
        print(f"✓ Date range: {start_date} to {end_date}")
    except Exception as e:
        print(f"✗ Date handling error: {e}")
        return False
    
    # Test 3: Test common tickers
    print("\n3. Testing common ticker symbols...")
    test_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "INVALID_TICKER"]
    
    for ticker in test_tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if data is not None and not data.empty and len(data) > 5:
                print(f"✓ {ticker}: {len(data)} data points")
            else:
                print(f"✗ {ticker}: No valid data")
        except Exception as e:
            print(f"✗ {ticker}: Error - {e}")
    
    # Test 4: Check data structure
    print("\n4. Testing data structure handling...")
    try:
        # Single ticker
        single_data = yf.download("AAPL", start=start_date, end=end_date, progress=False, auto_adjust=True)
        print(f"✓ Single ticker structure: {single_data.shape}")
        print(f"  Columns: {list(single_data.columns)}")
        
        # Multiple tickers
        multi_data = yf.download(["AAPL", "MSFT"], start=start_date, end=end_date, progress=False, auto_adjust=True)
        print(f"✓ Multiple ticker structure: {multi_data.shape}")
        print(f"  Column levels: {multi_data.columns.nlevels}")
        
    except Exception as e:
        print(f"✗ Data structure error: {e}")
        return False
    
    # Test 5: Check for NaN issues
    print("\n5. Testing NaN handling...")
    try:
        test_data = yf.download("AAPL", start=start_date, end=end_date, progress=False, auto_adjust=True)
        # Handle different data structures
        if isinstance(test_data.columns, pd.MultiIndex):
            # Multi-level columns (multiple tickers)
            close_prices = test_data[('Close', 'AAPL')]
        elif 'Close' in test_data.columns:
            close_prices = test_data['Close']
        else:
            close_prices = test_data.iloc[:, 0]  # First column
        
        returns = close_prices.pct_change().dropna()
        nan_count = returns.isna().sum()
        inf_count = np.isinf(returns).sum()
        
        print(f"✓ Returns calculated: {len(returns)} points")
        print(f"  NaN values: {nan_count}")
        print(f"  Infinite values: {inf_count}")
        
        if len(returns) > 0 and nan_count == 0 and inf_count == 0:
            print("✓ Data quality looks good")
        else:
            print("⚠ Data quality issues detected")
            
    except Exception as e:
        print(f"✗ Returns calculation error: {e}")
        return False
    
    # Test 6: Environment check
    print("\n6. Checking Python environment...")
    try:
        import matplotlib
        print(f"✓ matplotlib: {matplotlib.__version__}")
        
        import pandas
        print(f"✓ pandas: {pandas.__version__}")
        
        import numpy
        print(f"✓ numpy: {numpy.__version__}")
        
        import yfinance
        print(f"✓ yfinance: {yfinance.__version__}")
        
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        return False
    
    print("\n=== Diagnostic Complete ===")
    print("✓ All tests passed! The tool should work correctly.")
    return True

def suggest_solutions():
    """Provide solutions for common issues"""
    print("\n=== Common Issues and Solutions ===")
    print("\n1. 'Everything is NaN' issue:")
    print("   - Check internet connection")
    print("   - Verify ticker symbols are correct")
    print("   - Try different date ranges")
    print("   - Run: python diagnostic.py")
    
    print("\n2. 'No data loaded' issue:")
    print("   - Ensure tickers are valid (e.g., 'AAPL', not 'Apple')")
    print("   - Check if markets were open during date range")
    print("   - Try recent dates (last 1-2 years)")
    
    print("\n3. 'Module not found' errors:")
    print("   - Install requirements: pip install -r requirements.txt")
    print("   - Activate virtual environment if using one")
    
    print("\n4. 'Matplotlib backend' issues:")
    print("   - Set backend: import matplotlib; matplotlib.use('TkAgg')")
    print("   - Or try: matplotlib.use('Agg') for non-GUI environments")

if __name__ == "__main__":
    success = diagnose_data_issues()
    if not success:
        suggest_solutions()
    
    print(f"\nPython version: {sys.version}")
    print(f"Platform: {sys.platform}")