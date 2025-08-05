@echo off
echo Monte Carlo Simulation for all stocks
python main_app.py --tickers "AAPL,MSFT,GOOGL,TSLA" --analysis montecarlo --all-stocks
pause