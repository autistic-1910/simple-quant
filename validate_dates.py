#!/usr/bin/env python3
"""
Date validation utility
"""

from datetime import datetime, timedelta
import pandas as pd

def validate_and_suggest_dates(start_date, end_date):
    """Validate dates and suggest corrections"""
    
    print("=== Date Validation ===\n")
    
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        today = pd.Timestamp.now()
        
        print(f"Input dates:")
        print(f"  Start: {start_date}")
        print(f"  End: {end_date}")
        print(f"  Today: {today.strftime('%Y-%m-%d')}")
        
        issues = []
        
        # Check if start date is in the future
        if start > today:
            issues.append(f"Start date ({start_date}) is in the future")
        
        # Check if end date is in the future
        if end > today:
            issues.append(f"End date ({end_date}) is in the future")
        
        # Check if start is after end
        if start >= end:
            issues.append(f"Start date is not before end date")
        
        # Check if date range is too short
        if (end - start).days < 30:
            issues.append(f"Date range is very short ({(end - start).days} days)")
        
        if issues:
            print(f"\nIssues found:")
            for issue in issues:
                print(f"  - {issue}")
            
            # Suggest corrected dates
            suggested_end = min(end, today)
            suggested_start = max(start, suggested_end - timedelta(days=365*2))
            
            # Ensure start is not in the future
            if suggested_start > today:
                suggested_start = today - timedelta(days=365*2)
            
            print(f"\nSuggested corrections:")
            print(f"  Start: {suggested_start.strftime('%Y-%m-%d')}")
            print(f"  End: {suggested_end.strftime('%Y-%m-%d')}")
            print(f"  Range: {(suggested_end - suggested_start).days} days")
            
            return suggested_start.strftime('%Y-%m-%d'), suggested_end.strftime('%Y-%m-%d')
        
        else:
            print(f"\nDates are valid!")
            print(f"  Range: {(end - start).days} days")
            return start_date, end_date
            
    except Exception as e:
        print(f"\nError parsing dates: {e}")
        
        # Return safe default dates
        safe_end = datetime.now().strftime('%Y-%m-%d')
        safe_start = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        
        print(f"\nUsing safe default dates:")
        print(f"  Start: {safe_start}")
        print(f"  End: {safe_end}")
        
        return safe_start, safe_end

def get_market_calendar_info():
    """Provide information about market trading days"""
    
    print("\n=== Market Calendar Info ===")
    print("Stock markets are typically closed on:")
    print("  - Weekends (Saturday, Sunday)")
    print("  - Major holidays (New Year's Day, Christmas, etc.)")
    print("  - Market-specific holidays")
    print("\nFor best results:")
    print("  - Use weekdays when markets are open")
    print("  - Avoid major holiday periods")
    print("  - Allow for sufficient data points (at least 30 days)")

if __name__ == "__main__":
    # Test with your problematic dates
    print("Testing your problematic date range:")
    validate_and_suggest_dates("2024-01-01", "2025-01-01")
    
    print("\n" + "="*50)
    
    # Test with some other examples
    test_cases = [
        ("2023-01-01", "2024-01-01"),  # Good range
        ("2025-01-01", "2025-12-31"),  # Future dates
        ("2024-12-01", "2024-11-01"),  # Start after end
        ("2024-01-01", "2024-01-05"),  # Very short range
    ]
    
    for start, end in test_cases:
        print(f"\nTesting: {start} to {end}")
        validate_and_suggest_dates(start, end)
    
    get_market_calendar_info()