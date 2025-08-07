@echo off
echo Fixing your date range issue...
echo.
python fix_your_dates.py
echo.
echo Running analysis with corrected dates...
python main_app.py --tickers "AAPL,MSFT,GOOGL,TSLA" --analysis all --start "2023-08-07" --end "2024-12-31"
echo.
pause