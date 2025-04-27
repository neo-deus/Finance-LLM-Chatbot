import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

class Visualizer:
    """Helper class for creating finance-related visualizations"""
    
    @staticmethod
    def plot_stock(ticker, data, period="1y"):
        """
        Create and save a plot of the stock price
        
        Args:
            ticker (str): Stock ticker symbol
            data (pandas.DataFrame): Stock price data
            period (str): Time period displayed
            
        Returns:
            str: Path to the saved plot
        """
        if data is None or data.empty:
            return None
        
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Close'], label=f"{ticker} Close Price")
        
        # Add moving averages
        if 'MA50' not in data.columns:
            data['MA50'] = data['Close'].rolling(window=50).mean()
        if 'MA200' not in data.columns:
            data['MA200'] = data['Close'].rolling(window=200).mean()
            
        plt.plot(data.index, data['MA50'], label="50-day MA", alpha=0.7)
        plt.plot(data.index, data['MA200'], label="200-day MA", alpha=0.7)
        
        plt.title(f"{ticker} Stock Price ({period})")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plot_path = f"{ticker}_stock_plot.png"
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    @staticmethod
    def plot_portfolio_composition(allocation):
        """
        Create a pie chart of portfolio composition
        
        Args:
            allocation (dict): Portfolio allocation data
            
        Returns:
            str: Path to saved plot file
        """
        if not allocation:
            return None
            
        # Extract labels and sizes
        labels = list(allocation.keys())
        values = [item["value"] for item in allocation.values()]
        percentages = [item["percentage"] for item in allocation.values()]
        
        # Create custom labels with percentages
        custom_labels = [f"{label} (${value:.2f}, {pct:.1f}%)" 
                        for label, value, pct in zip(labels, values, percentages)]
        
        # Create pie chart
        plt.figure(figsize=(10, 8))
        plt.pie(values, labels=custom_labels, autopct=lambda pct: f"{pct:.1f}%", 
                startangle=90, shadow=True)
        plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
        plt.title("Portfolio Composition")
        
        # Save plot
        plot_path = "portfolio_composition.png"
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    @staticmethod
    def plot_portfolio_performance(dates, values, period="1y"):
        """
        Create a plot of portfolio performance
        
        Args:
            dates (list): List of date strings
            values (list): List of portfolio values
            period (str): Time period for the plot
            
        Returns:
            str: Path to saved plot file
        """
        if len(dates) == 0 or len(values) == 0:
            return None
        
        # Convert dates to datetime objects if they are strings
        if isinstance(dates[0], str):
            dates = [datetime.strptime(date, "%Y-%m-%d") for date in dates]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(dates, values)
        plt.title(f"Portfolio Performance ({period})")
        plt.xlabel("Date")
        plt.ylabel("Value ($)")
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = "portfolio_performance.png"
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    @staticmethod
    def plot_stock_comparison(data1, data2, ticker1, ticker2, period="1y"):
        """
        Create a comparison plot of two stocks
        
        Args:
            data1 (pandas.DataFrame): First stock data
            data2 (pandas.DataFrame): Second stock data
            ticker1 (str): First ticker symbol
            ticker2 (str): Second ticker symbol
            period (str): Time period for the plot
            
        Returns:
            str: Path to saved plot file
        """
        if data1 is None or data1.empty or data2 is None or data2.empty:
            return None
        
        # Normalize data for comparison (starting value = 100)
        normalized1 = data1['Close'] / data1['Close'].iloc[0] * 100
        normalized2 = data2['Close'] / data2['Close'].iloc[0] * 100
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(normalized1.index, normalized1, label=ticker1)
        plt.plot(normalized2.index, normalized2, label=ticker2)
        plt.title(f"Comparison: {ticker1} vs {ticker2} ({period})")
        plt.xlabel("Date")
        plt.ylabel("Normalized Price (Start=100)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = f"{ticker1}_vs_{ticker2}_comparison.png"
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path