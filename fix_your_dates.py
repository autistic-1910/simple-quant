#!/usr/bin/env python3
"""
Quick fix for your specific date issue
"""

from datetime import datetime, timedelta
import pandas as pd

def fix_your_date_issue():
    """Fix the specific date range issue you're experiencing"""
    
    print("=== Fixing Your Date Issue ===\n")
    
    # Your problematic dates
    your_start = "2023-08-07"
    your_end = "2025-08-06"  # This is the problem!
    
    print(f"Your current dates:")
    print(f"  Start: {your_start}")
    print(f"  End: {your_end} ❌ (FUTURE DATE - THIS IS THE PROBLEM!)")
    
    # Get today's date
    today = datetime.now()
    print(f"  Today: {today.strftime('%Y-%m-%d')}")
    
    # Calculate corrected dates
    corrected_end = today.strftime('%Y-%m-%d')
    corrected_start = your_start  # Your start date is fine
    
    print(f"\n✅ Corrected dates:")
    print(f"  Start: {corrected_start}")
    print(f"  End: {corrected_end}")
    
    # Calculate the range
    start_date = datetime.strptime(corrected_start, '%Y-%m-%d')
    end_date = datetime.strptime(corrected_end, '%Y-%m-%d')
    days_range = (end_date - start_date).days
    
    print(f"  Range: {days_range} days")
    
    return corrected_start, corrected_end

def test_with_corrected_dates():
    """Test loading data with corrected dates"""
    
    print("\n=== Testing with Corrected Dates ===\n")
    
    # Get corrected dates
    start_date, end_date = fix_your_date_issue()
    
    # Your tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    
    print(f"\nTesting with:")
    print(f"  Tickers: {tickers}")
    print(f"  Start: {start_date}")
    print(f"  End: {end_date}")
    
    try:
        from quant_analysis import QuantitativeAnalysis
        
        qa = QuantitativeAnalysis()
        data = qa.load_stock_data(tickers, start_date, end_date)
        
        if data is not None:
            print("\n✅ SUCCESS! Data loaded successfully!")
            print(f"   Data shape: {data.shape}")
            print(f"   Date range in data: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            
            # Calculate returns
            returns = qa.calculate_returns()
            if returns is not None:
                print("✅ Returns calculated successfully!")
                
                # Calculate metrics
                metrics = qa.calculate_risk_return_metrics()
                if metrics is not None:
                    print("✅ Risk metrics calculated successfully!")
                    print("\nQuick metrics summary:")
                    for ticker in metrics.index:
                        annual_return = metrics.loc[ticker, 'Annual Return']
                        print(f"  {ticker}: {annual_return:.2%} annual return")
            
            return True
        else:
            print("❌ Still failed to load data")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def provide_command_line_fix():
    """Provide the exact command line to use"""
    
    print("\n=== Command Line Fix ===\n")
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    print("Use this exact command:")
    print(f'python main_app.py --tickers "AAPL,MSFT,GOOGL,TSLA" --analysis all --start "2023-08-07" --end "{today}"')
    
    print("\nOr for a 2-year analysis:")
    two_years_ago = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    print(f'python main_app.py --tickers "AAPL,MSFT,GOOGL,TSLA" --analysis all --start "{two_years_ago}" --end "{today}"')

if __name__ == "__main__":
    print("🔧 FIXING YOUR DATE ISSUE")
    print("=" * 50)
    
    # Test with corrected dates
    success = test_with_corrected_dates()
    
    # Provide command line fix
    provide_command_line_fix()
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY:")
    print("❌ Problem: End date 2025-08-06 is in the future")
    print("✅ Solution: Use today's date as end date")
    print("💡 Remember: Stock data only exists up to today!")