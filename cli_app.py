import argparse
import sys
from quant_analysis import QuantitativeAnalysis
import config

def main():
    parser = argparse.ArgumentParser(description='Quantitative Finance Analysis Tool')
    parser.add_argument('--tickers', type=str, default='AAPL,MSFT', 
                       help='Comma-separated list of stock tickers')
    parser.add_argument('--start', type=str, default=config.START_DATE,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=config.END_DATE,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--analysis', type=str, choices=['basic', 'portfolio', 'montecarlo', 'all'],
                       default='all', help='Type of analysis to perform')
    parser.add_argument('--export', type=str, help='Export results to file (CSV or Excel)')
    
    args = parser.parse_args()
    
    # Initialize analysis
    quant = QuantitativeAnalysis()
    
    # Parse tickers
    tickers = [ticker.strip() for ticker in args.tickers.split(',')]
    
    print(f"Loading data for: {tickers}")
    print(f"Date range: {args.start} to {args.end}")
    
    # Load data
    data = quant.load_stock_data(tickers, args.start, args.end)
    if data is None:
        print("Failed to load data!")
        sys.exit(1)
    
    # Calculate returns
    returns = quant.calculate_returns()
    print(f"Returns calculated for {len(returns.columns)} assets")
    
    # Perform analysis based on user choice
    if args.analysis in ['basic', 'all']:
        print("\n=== Risk and Return Metrics ===")
        metrics = quant.calculate_risk_return_metrics()
        print(metrics.round(4))
    
    if args.analysis in ['portfolio', 'all']:
        print("\n=== Portfolio Analysis ===")
        portfolio_result = quant.portfolio_analysis()
        if portfolio_result is not None:
            portfolio_metrics, portfolio_returns = portfolio_result
            if portfolio_metrics:
                for key, value in portfolio_metrics.items():
                    print(f"{key}: {value:.4f}")
    
    if args.analysis in ['montecarlo', 'all']:
        print("\n=== Monte Carlo Simulation ===")
        mc_results = quant.monte_carlo_simulation()
        quant.plot_monte_carlo(mc_results, tickers[0])
    
    if args.analysis == 'all':
        print("\n=== Correlation Analysis ===")
        correlation = quant.correlation_analysis()
        
        if len(tickers) > 1:
            print("\n=== Efficient Frontier ===")
            ef_results = quant.efficient_frontier()
            if ef_results:
                for key, value in ef_results.items():
                    print(f"{key}: {value:.4f}")
    
    # Export results if requested
    if args.export:
        try:
            if args.export.endswith('.xlsx'):
                import pandas as pd
                with pd.ExcelWriter(args.export) as writer:
                    returns.to_excel(writer, sheet_name='Returns')
                    metrics = quant.calculate_risk_return_metrics()
                    metrics.to_excel(writer, sheet_name='Metrics')
            else:
                returns.to_csv(args.export)
            print(f"\nResults exported to {args.export}")
        except Exception as e:
            print(f"Error exporting: {e}")

if __name__ == "__main__":
    main()