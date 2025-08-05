import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from quant_analysis import QuantitativeAnalysis
import config

class QuantAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Quantitative Finance Analysis Tool")
        self.root.geometry("1200x800")
        
        self.quant = QuantitativeAnalysis()
        self.setup_ui()
        
    def setup_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="Input Parameters", padding="10")
        input_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
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
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Load Data", command=self.load_data).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Calculate Metrics", command=self.calculate_metrics).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Show Correlation", command=self.show_correlation).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Monte Carlo", command=self.monte_carlo).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Efficient Frontier", command=self.efficient_frontier).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Export Results", command=self.export_results).pack(side=tk.LEFT, padx=(0, 10))
        
        # Results display
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Text results tab
        self.text_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.text_frame, text="Results")
        
        self.results_text = tk.Text(self.text_frame, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(self.text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Chart tab
        self.chart_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.chart_frame, text="Charts")
        
    def load_data(self):
        try:
            tickers = [ticker.strip() for ticker in self.tickers_var.get().split(',')]
            start_date = self.start_date_var.get()
            end_date = self.end_date_var.get()
            
            self.results_text.delete(1.0, tk.END)
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
            else:
                self.results_text.insert(tk.END, "Failed to load data!\n")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error loading data: {str(e)}")
    
    def calculate_metrics(self):
        try:
            if self.quant.returns is None:
                messagebox.showwarning("Warning", "Please load data first!")
                return
            
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
            
        except Exception as e:
            messagebox.showerror("Error", f"Error calculating metrics: {str(e)}")
    
    def show_correlation(self):
        try:
            if self.quant.returns is None:
                messagebox.showwarning("Warning", "Please load data first!")
                return
            
            correlation_matrix = self.quant.correlation_analysis()
            
            self.results_text.insert(tk.END, "=== Correlation Matrix ===\n")
            self.results_text.insert(tk.END, correlation_matrix.to_string())
            self.results_text.insert(tk.END, "\n\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error showing correlation: {str(e)}")
    
    def monte_carlo(self):
        try:
            if self.quant.returns is None:
                messagebox.showwarning("Warning", "Please load data first!")
                return
            
            ticker = self.quant.returns.columns[0]
            mc_results = self.quant.monte_carlo_simulation(days=252, simulations=1000, ticker=ticker)
            self.quant.plot_monte_carlo(mc_results, ticker)
            
            self.results_text.insert(tk.END, f"Monte Carlo simulation completed for {ticker}\n")
            self.results_text.insert(tk.END, f"Simulated {mc_results.shape[1]} paths for {mc_results.shape[0]} days\n\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error running Monte Carlo: {str(e)}")
    
    def efficient_frontier(self):
        try:
            if self.quant.returns is None:
                messagebox.showwarning("Warning", "Please load data first!")
                return
            
            if len(self.quant.returns.columns) < 2:
                messagebox.showwarning("Warning", "Need at least 2 assets for efficient frontier!")
                return
            
            results = self.quant.efficient_frontier()
            
            if results:
                self.results_text.insert(tk.END, "=== Efficient Frontier Analysis ===\n")
                for key, value in results.items():
                    self.results_text.insert(tk.END, f"{key}: {value:.4f}\n")
                self.results_text.insert(tk.END, "\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error calculating efficient frontier: {str(e)}")
    
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
                if filename.endswith('.xlsx'):
                    with pd.ExcelWriter(filename) as writer:
                        self.quant.returns.to_excel(writer, sheet_name='Returns')
                        metrics = self.quant.calculate_risk_return_metrics()
                        metrics.to_excel(writer, sheet_name='Metrics')
                        correlation = self.quant.returns.corr()
                        correlation.to_excel(writer, sheet_name='Correlation')
                else:
                    self.quant.returns.to_csv(filename)
                
                messagebox.showinfo("Success", f"Results exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting results: {str(e)}")

def main():
    root = tk.Tk()
    app = QuantAnalysisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()