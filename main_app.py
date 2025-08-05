import argparse
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from quant_analysis import QuantitativeAnalysis
import config

class QuantAnalysisApp:
    def __init__(self):
        self.quant = QuantitativeAnalysis()
        
    def run_cli(self, args):
        """Run command line interface"""
        # Parse tickers
        tickers = [ticker.strip() for ticker in args.tickers.split(',')]
        
        print(f"Loading data for: {tickers}")
        print(f"Date range: {args.start} to {args.end}")
        
        # Load data
        data = self.quant.load_stock_data(tickers, args.start, args.end)
        if data is None:
            print("Failed to load data!")
            sys.exit(1)
        
        # Calculate returns
        returns = self.quant.calculate_returns()
        print(f"Returns calculated for {len(returns.columns)} assets")
        
        # Perform analysis based on user choice
        if args.analysis in ['basic', 'all']:
            print("\n=== Risk and Return Metrics ===")
            metrics = self.quant.calculate_risk_return_metrics()
            print(metrics.round(4))
        
        if args.analysis in ['portfolio', 'all']:
            print("\n=== Portfolio Analysis ===")
            portfolio_metrics = self.quant.portfolio_analysis()
            for key, value in portfolio_metrics.items():
                if key != 'Portfolio Returns':
                    print(f"{key}: {value:.4f}")
        
        if args.analysis in ['montecarlo', 'all']:
            print("\n=== Monte Carlo Simulation ===")
            if args.all_stocks:
                mc_results_all = self.quant.monte_carlo_all_stocks(days=args.days, simulations=args.simulations)
            else:
                mc_results = self.quant.monte_carlo_simulation(days=args.days, simulations=args.simulations, ticker=tickers[0])
                self.quant.plot_monte_carlo(mc_results, tickers[0])
        
        if args.analysis == 'all':
            print("\n=== Correlation Analysis ===")
            correlation = self.quant.correlation_analysis()
            
            if len(tickers) > 1:
                print("\n=== Efficient Frontier ===")
                ef_results = self.quant.efficient_frontier()
                if ef_results:
                    for key, value in ef_results.items():
                        print(f"{key}: {value:.4f}")
        
        # Export results if requested
        if args.export:
            try:
                if args.export.endswith('.xlsx'):
                    with pd.ExcelWriter(args.export) as writer:
                        returns.to_excel(writer, sheet_name='Returns')
                        metrics = self.quant.calculate_risk_return_metrics()
                        metrics.to_excel(writer, sheet_name='Metrics')
                        if len(tickers) > 1:
                            correlation = returns.corr()
                            correlation.to_excel(writer, sheet_name='Correlation')
                else:
                    returns.to_csv(args.export)
                print(f"\nResults exported to {args.export}")
            except Exception as e:
                print(f"Error exporting: {e}")

    def run_gui(self):
        """Run graphical user interface"""
        root = tk.Tk()
        gui = QuantAnalysisGUI(root, self.quant)
        root.mainloop()

class QuantAnalysisGUI:
    def __init__(self, root, quant_instance):
        self.root = root
        self.root.title("Quantitative Finance Analysis Tool")
        self.root.geometry("1200x800")
        
        self.quant = quant_instance
        self.setup_ui()
        
    def setup_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Quantitative Finance Analysis Tool", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="Input Parameters", padding="10")
        input_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Tickers input
        ttk.Label(input_frame, text="Tickers (comma-separated):").grid(row=0, column=0, sticky=tk.W)
        self.tickers_var = tk.StringVar(value="AAPL,MSFT,GOOGL,TSLA")
        ttk.Entry(input_frame, textvariable=self.tickers_var, width=40).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        
        # Date inputs
        ttk.Label(input_frame, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.start_date_var = tk.StringVar(value=config.START_DATE)
        ttk.Entry(input_frame, textvariable=self.start_date_var, width=20).grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        ttk.Label(input_frame, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.end_date_var = tk.StringVar(value=config.END_DATE)
        ttk.Entry(input_frame, textvariable=self.end_date_var, width=20).grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        # Monte Carlo parameters
        ttk.Label(input_frame, text="Monte Carlo Simulations:").grid(row=3, column=0, sticky=tk.W, pady=(10, 0))
        self.simulations_var = tk.StringVar(value="1000")
        ttk.Entry(input_frame, textvariable=self.simulations_var, width=20).grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        ttk.Label(input_frame, text="Forecast Days:").grid(row=4, column=0, sticky=tk.W, pady=(10, 0))
        self.days_var = tk.StringVar(value="252")
        ttk.Entry(input_frame, textvariable=self.days_var, width=20).grid(row=4, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Row 1 buttons
        button_row1 = ttk.Frame(button_frame)
        button_row1.pack(pady=(0, 5))
        
        ttk.Button(button_row1, text="Load Data", command=self.load_data).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_row1, text="Calculate Metrics", command=self.calculate_metrics).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_row1, text="Show Correlation", command=self.show_correlation).pack(side=tk.LEFT, padx=(0, 10))
        
        # Row 2 buttons
        button_row2 = ttk.Frame(button_frame)
        button_row2.pack(pady=(0, 5))
        
        ttk.Button(button_row2, text="Monte Carlo (Single)", command=self.monte_carlo_single).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_row2, text="Monte Carlo (All)", command=self.monte_carlo_all).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_row2, text="Efficient Frontier", command=self.efficient_frontier).pack(side=tk.LEFT, padx=(0, 10))
        
        # Row 3 buttons
        button_row3 = ttk.Frame(button_frame)
        button_row3.pack()
        
        ttk.Button(button_row3, text="Run Full Analysis", command=self.run_full_analysis).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_row3, text="Export Results", command=self.export_results).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_row3, text="Clear Results", command=self.clear_results).pack(side=tk.LEFT, padx=(0, 10))
        
        # Results display
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Text results tab
        self.text_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.text_frame, text="Results")
        
        self.results_text = tk.Text(self.text_frame, wrap=tk.WORD, font=('Consolas', 10))
        scrollbar = ttk.Scrollbar(self.text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Chart tab
        self.chart_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.chart_frame, text="Charts")
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def update_status(self, message):
        """Update status bar"""
        self.status_var.set(message)
        self.root.update()
        
    def load_data(self):
        try:
            self.update_status("Loading data...")
            tickers = [ticker.strip() for ticker in self.tickers_var.get().split(',')]
            start_date = self.start_date_var.get()
            end_date = self.end_date_var.get()
            
            self.results_text.insert(tk.END, f"Loading data for: {tickers}\n")
            self.results_text.insert(tk.END, f"Date range: {start_date} to {end_date}\n\n")
            
            data = self.quant.load_stock_data(tickers, start_date, end_date)
            
            if data is not None:
                self.results_text.insert(tk.END, "Data loaded successfully!\n")
                self.results_text.insert(tk.END, f"Data shape: {data.shape}\n\n")
                
                # Calculate and display returns
                returns = self.quant.calculate_returns()
                self.results_text.insert(tk.END, "Returns calculated.\n")
                self.results_text.insert(tk.END, f"Returns shape: {returns.shape}\n\n")
                self.update_status("Data loaded successfully")
            else:
                self.results_text.insert(tk.END, "Failed to load data!\n")
                self.update_status("Failed to load data")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error loading data: {str(e)}")
            self.update_status("Error loading data")
    
    def calculate_metrics(self):
        try:
            if self.quant.returns is None:
                messagebox.showwarning("Warning", "Please load data first!")
                return
            
            self.update_status("Calculating metrics...")
            metrics = self.quant.calculate_risk_return_metrics()
            
            self.results_text.insert(tk.END, "=== Risk and Return Metrics ===\n")
            self.results_text.insert(tk.END, metrics.to_string())
            self.results_text.insert(tk.END, "\n\n")
            
            # Portfolio analysis
            portfolio_metrics = self.quant.portfolio_analysis()
            self.results_text.insert(tk.END, "=== Portfolio Analysis (Equal Weights) ===\n")
            for key, value in portfolio_metrics.items():
                if key != 'Portfolio Returns':
                    self.results_text.insert(tk.END, f"{key}: {value:.4f}\n")
            self.results_text.insert(tk.END, "\n")
            self.update_status("Metrics calculated")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error calculating metrics: {str(e)}")
            self.update_status("Error calculating metrics")
    
    def show_correlation(self):
        try:
            if self.quant.returns is None:
                messagebox.showwarning("Warning", "Please load data first!")
                return
            
            self.update_status("Generating correlation analysis...")
            correlation_matrix = self.quant.correlation_analysis()
            
            self.results_text.insert(tk.END, "=== Correlation Matrix ===\n")
            self.results_text.insert(tk.END, correlation_matrix.to_string())
            self.results_text.insert(tk.END, "\n\n")
            self.update_status("Correlation analysis complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error showing correlation: {str(e)}")
            self.update_status("Error in correlation analysis")
    
    def monte_carlo_single(self):
        try:
            if self.quant.returns is None:
                messagebox.showwarning("Warning", "Please load data first!")
                return
            
            self.update_status("Running Monte Carlo simulation...")
            ticker = self.quant.returns.columns[0]
            days = int(self.days_var.get())
            simulations = int(self.simulations_var.get())
            
            mc_results = self.quant.monte_carlo_simulation(days=days, simulations=simulations, ticker=ticker)
            self.quant.plot_monte_carlo(mc_results, ticker)
            
            self.results_text.insert(tk.END, f"Monte Carlo simulation completed for {ticker}\n")
            self.results_text.insert(tk.END, f"Simulated {mc_results.shape[1]} paths for {mc_results.shape[0]} days\n\n")
            self.update_status("Monte Carlo simulation complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error running Monte Carlo: {str(e)}")
            self.update_status("Error in Monte Carlo simulation")
    
    def monte_carlo_all(self):
        try:
            if self.quant.returns is None:
                messagebox.showwarning("Warning", "Please load data first!")
                return
            
            self.update_status("Running Monte Carlo for all stocks...")
            days = int(self.days_var.get())
            simulations = int(self.simulations_var.get())
            
            mc_results_all = self.quant.monte_carlo_all_stocks(days=days, simulations=simulations)
            
            self.results_text.insert(tk.END, f"Monte Carlo simulation completed for all stocks\n")
            self.results_text.insert(tk.END, f"Simulated {simulations} paths for {days} days each\n\n")
            self.update_status("Monte Carlo simulation complete for all stocks")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error running Monte Carlo: {str(e)}")
            self.update_status("Error in Monte Carlo simulation")
    
    def efficient_frontier(self):
        try:
            if self.quant.returns is None:
                messagebox.showwarning("Warning", "Please load data first!")
                return
            
            if len(self.quant.returns.columns) < 2:
                messagebox.showwarning("Warning", "Need at least 2 assets for efficient frontier!")
                return
            
            self.update_status("Calculating efficient frontier...")
            results = self.quant.efficient_frontier()
            
            if results:
                self.results_text.insert(tk.END, "=== Efficient Frontier Analysis ===\n")
                for key, value in results.items():
                    self.results_text.insert(tk.END, f"{key}: {value:.4f}\n")
                self.results_text.insert(tk.END, "\n")
            self.update_status("Efficient frontier analysis complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error calculating efficient frontier: {str(e)}")
            self.update_status("Error in efficient frontier analysis")
    
    def run_full_analysis(self):
        """Run complete analysis workflow"""
        try:
            self.update_status("Running full analysis...")
            
            # Load data if not already loaded
            if self.quant.returns is None:
                self.load_data()
                if self.quant.returns is None:
                    return
            
            # Run all analyses
            self.results_text.insert(tk.END, "\n" + "="*60 + "\n")
            self.results_text.insert(tk.END, "FULL QUANTITATIVE ANALYSIS REPORT\n")
            self.results_text.insert(tk.END, "="*60 + "\n\n")
            
            self.calculate_metrics()
            self.show_correlation()
            
            # Price performance
            self.results_text.insert(tk.END, "=== Price Performance Chart ===\n")
            self.quant.plot_price_performance()
            self.results_text.insert(tk.END, "Price performance chart displayed\n\n")
            
            # Monte Carlo for first stock
            self.monte_carlo_single()
            
            # Efficient frontier if multiple stocks
            if len(self.quant.returns.columns) > 1:
                self.efficient_frontier()
            
            self.results_text.insert(tk.END, "\n" + "="*60 + "\n")
            self.results_text.insert(tk.END, "ANALYSIS COMPLETE\n")
            self.results_text.insert(tk.END, "="*60 + "\n\n")
            
            self.update_status("Full analysis complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error in full analysis: {str(e)}")
            self.update_status("Error in full analysis")
    
    def clear_results(self):
        """Clear the results text area"""
        self.results_text.delete(1.0, tk.END)
        self.update_status("Results cleared")
    
    def export_results(self):
        try:
            if self.quant.returns is None:
                messagebox.showwarning("Warning", "No data to export!")
                return
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")]
            )
            
            if filename:
                self.update_status("Exporting results...")
                if filename.endswith('.xlsx'):
                    with pd.ExcelWriter(filename) as writer:
                        self.quant.returns.to_excel(writer, sheet_name='Returns')
                        metrics = self.quant.calculate_risk_return_metrics()
                        metrics.to_excel(writer, sheet_name='Metrics')
                        correlation = self.quant.returns.corr()
                        correlation.to_excel(writer, sheet_name='Correlation')
                        
                        # Add text results
                        text_results = self.results_text.get(1.0, tk.END)
                        text_df = pd.DataFrame({'Analysis Results': [text_results]})
                        text_df.to_excel(writer, sheet_name='Analysis_Report', index=False)
                else:
                    self.quant.returns.to_csv(filename)
                
                messagebox.showinfo("Success", f"Results exported to {filename}")
                self.update_status(f"Results exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting results: {str(e)}")
            self.update_status("Error exporting results")

def main():
    """Main function to handle both CLI and GUI modes"""
    app = QuantAnalysisApp()
    
    # Check if any command line arguments are provided
    if len(sys.argv) > 1:
        # CLI mode
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
        parser.add_argument('--simulations', type=int, default=1000, 
                           help='Number of Monte Carlo simulations')
        parser.add_argument('--days', type=int, default=252, 
                           help='Number of days to forecast')
        parser.add_argument('--all-stocks', action='store_true', 
                           help='Run Monte Carlo for all stocks')
        parser.add_argument('--gui', action='store_true', 
                           help='Force GUI mode even with other arguments')
        
        args = parser.parse_args()
        
        if args.gui:
            # Force GUI mode
            app.run_gui()
        else:
            # CLI mode
            app.run_cli(args)
    else:
        # No arguments provided, default to GUI mode
        print("No command line arguments provided. Starting GUI mode...")
        print("Use --help to see command line options.")
        app.run_gui()

if __name__ == "__main__":
    main()