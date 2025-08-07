#!/usr/bin/env python3
"""
Immediate troubleshooting for your specific issue
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import traceback

def test_basic_connectivity():
    """Test basic yfinance connectivity"""
    print("=== STEP 1: Testing Basic Connectivity ===")
    
    try:
        print("Testing simple AAPL download...")
        data = yf.download("AAPL", start="2024-01-01", end="2024-01-10", progress=False)
        
        if data is not None and not data.empty:
            print(f"✅ SUCCESS: Downloaded {len(data)} rows")
            print(f"   Columns: {list(data.columns)}")
            print(f"   Date range: {data.index[0]} to {data.index[-1]}")
            return True
        else:
            print("❌ FAILED: No data returned")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print("Full error:")
        traceback.print_exc()
        return False

def test_your_exact_tickers():
    """Test your exact tickers with safe dates"""
    print("\n=== STEP 2: Testing Your Exact Tickers ===")
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    
    # Use safe dates
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f"Using safe dates: {start_date} to {end_date}")
    print(f"Testing tickers: {tickers}")
    
    success_count = 0
    
    for ticker in tickers:
        try:
            print(f"\nTesting {ticker}...")
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data is not None and not data.empty:
                print(f"  ✅ {ticker}: {len(data)} data points")
                success_count += 1
            else:
                print(f"  ❌ {ticker}: No data")
                
        except Exception as e:
            print(f"  ❌ {ticker}: Error - {e}")
    
    print(f"\nResult: {success_count}/{len(tickers)} tickers successful")
    return success_count == len(tickers)

def test_multiple_tickers_together():
    """Test downloading multiple tickers together"""
    print("\n=== STEP 3: Testing Multiple Tickers Together ===")
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    try:
        print(f"Downloading {tickers} together...")
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        if data is not None and not data.empty:
            print(f"✅ SUCCESS: Shape {data.shape}")
            print(f"   Column structure: {data.columns.nlevels} levels")
            print(f"   Date range: {data.index[0]} to {data.index[-1]}")
            
            # Check for Close prices
            if 'Close' in data.columns:
                close_data = data['Close']
                print(f"   Close prices shape: {close_data.shape}")
                print(f"   Close columns: {list(close_data.columns)}")
                return True
            else:
                print("   ❌ No 'Close' column found")
                return False
        else:
            print("❌ FAILED: No data returned")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        traceback.print_exc()
        return False

def test_quant_analysis_class():
    """Test the QuantitativeAnalysis class directly"""
    print("\n=== STEP 4: Testing QuantitativeAnalysis Class ===")
    
    try:
        from quant_analysis import QuantitativeAnalysis
        print("✅ Successfully imported QuantitativeAnalysis")
        
        qa = QuantitativeAnalysis()
        print("✅ Successfully created instance")
        
        # Test with safe dates
        tickers = ['AAPL', 'MSFT']
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        print(f"Testing load_stock_data with {tickers}")
        print(f"Date range: {start_date} to {end_date}")
        
        data = qa.load_stock_data(tickers, start_date, end_date)
        
        if data is not None:
            print(f"✅ SUCCESS: Data loaded, shape {data.shape}")
            
            # Test returns calculation
            returns = qa.calculate_returns()
            if returns is not None:
                print(f"✅ Returns calculated, shape {returns.shape}")
                return True
            else:
                print("❌ Failed to calculate returns")
                return False
        else:
            print("❌ Failed to load data")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        traceback.print_exc()
        return False

def check_internet_and_firewall():
    """Check internet connectivity and potential firewall issues"""
    print("\n=== STEP 5: Checking Internet and Firewall ===")
    
    try:
        import urllib.request
        import socket
        
        # Test basic internet
        print("Testing internet connectivity...")
        response = urllib.request.urlopen('https://www.google.com', timeout=10)
        print("✅ Internet connection working")
        
        # Test Yahoo Finance specifically
        print("Testing Yahoo Finance connectivity...")
        response = urllib.request.urlopen('https://finance.yahoo.com', timeout=10)
        print("✅ Yahoo Finance accessible")
        
        return True
        
    except Exception as e:
        print(f"❌ Connectivity issue: {e}")
        print("This might be a firewall or network issue")
        return False

def provide_immediate_solutions():
    """Provide immediate solutions based on test results"""
    print("\n" + "="*60)
    print("🔧 IMMEDIATE SOLUTIONS")
    print("="*60)
    
    print("\n1. Try this exact command right now:")
    today = datetime.now().strftime('%Y-%m-%d')
    one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f'   python main_app.py --tickers "AAPL,MSFT" --analysis basic --start "{one_year_ago}" --end "{today}"')
    
    print("\n2. If that fails, try individual ticker:")
    print(f'   python main_app.py --tickers "AAPL" --analysis basic --start "{one_year_ago}" --end "{today}"')
    
    print("\n3. Run diagnostic:")
    print("   python diagnostic.py")
    
    print("\n4. Check your virtual environment:")
    print("   Make sure you're in the right environment")
    print("   Try: pip install --upgrade yfinance")

if __name__ == "__main__":
    print("🚨 TROUBLESHOOTING YOUR DATA LOADING ISSUE")
    print("="*60)
    
    # Run all tests
    test1 = test_basic_connectivity()
    test2 = test_your_exact_tickers() if test1 else False
    test3 = test_multiple_tickers_together() if test2 else False
    test4 = test_quant_analysis_class() if test3 else False
    test5 = check_internet_and_firewall()
    
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    print(f"1. Basic connectivity: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"2. Individual tickers: {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"3. Multiple tickers: {'✅ PASS' if test3 else '❌ FAIL'}")
    print(f"4. QuantAnalysis class: {'✅ PASS' if test4 else '❌ FAIL'}")
    print(f"5. Internet/Firewall: {'✅ PASS' if test5 else '❌ FAIL'}")
    
    if all([test1, test2, test3, test4, test5]):
        print("\n🎉 ALL TESTS PASSED! The issue might be with your specific command or dates.")
    else:
        print("\n⚠️  ISSUES DETECTED! See solutions below.")
    
    provide_immediate_solutions()
    
    print(f"\nSystem Info:")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")