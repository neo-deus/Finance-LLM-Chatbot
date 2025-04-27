import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from modules.stock_data import StockDataService
from modules.visualization import Visualizer
from modules.config import Config

class Portfolio:
    """
    Class for analyzing and managing stock portfolios.
    Provides methods for portfolio valuation, performance analysis, and risk assessment.
    """
    
    def __init__(self, portfolio_file=Config.PORTFOLIO_FILE):
        """
        Initialize the Portfolio
        
        Args:
            portfolio_file (str): Path to portfolio JSON file
        """
        self.portfolio_file = portfolio_file
        self.portfolio = self._load_portfolio()
        self.stock_data_service = StockDataService()
        
    def _load_portfolio(self):
        """
        Load portfolio data from JSON file
        
        Returns:
            dict: Portfolio data
        """
        try:
            with open(self.portfolio_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Portfolio file {self.portfolio_file} not found. Creating empty portfolio.")
            return {"stocks": {}, "cash": 0.0, "transactions": []}
        except json.JSONDecodeError:
            print(f"Error parsing portfolio file {self.portfolio_file}. Creating empty portfolio.")
            return {"stocks": {}, "cash": 0.0, "transactions": []}
    
    def save_portfolio(self):
        """Save the current portfolio to the JSON file"""
        with open(self.portfolio_file, 'w') as f:
            json.dump(self.portfolio, f, indent=4)
    
    def get_portfolio_value(self):
        """
        Calculate the current total value of the portfolio
        
        Returns:
            float: Total portfolio value
        """
        total_value = self.portfolio.get("cash", 0)
        
        for ticker, position in self.portfolio.get("stocks", {}).items():
            shares = position.get("shares", 0)
            if shares <= 0:
                continue
                
            data = self.stock_data_service.get_stock_data(ticker)
            if data is not None and not data.empty:
                current_price = data['Close'].iloc[-1]
                position_value = shares * current_price
                total_value += position_value
        
        return total_value
    
    def get_portfolio_allocation(self):
        """
        Calculate the current allocation of the portfolio
        
        Returns:
            dict: Portfolio allocation by ticker
        """
        allocation = {}
        total_value = self.get_portfolio_value()
        
        if total_value <= 0:
            return allocation
        
        # Add cash allocation
        cash_value = self.portfolio.get("cash", 0)
        allocation["Cash"] = {
            "value": cash_value,
            "percentage": (cash_value / total_value) * 100
        }
        
        # Add stock allocations
        for ticker, position in self.portfolio.get("stocks", {}).items():
            shares = position.get("shares", 0)
            if shares <= 0:
                continue
                
            data = self.stock_data_service.get_stock_data(ticker)
            if data is not None and not data.empty:
                current_price = data['Close'].iloc[-1]
                position_value = shares * current_price
                allocation[ticker] = {
                    "value": position_value,
                    "percentage": (position_value / total_value) * 100
                }
        
        return allocation
    
    def calculate_portfolio_performance(self, period="1y"):
        """
        Calculate the performance of the portfolio over a given period
        
        Args:
            period (str): Time period for performance calculation
            
        Returns:
            dict: Portfolio performance metrics
        """
        # Calculate current value
        current_value = self.get_portfolio_value()
        
        # Calculate historical value
        historical_values = []
        dates = []
        
        # Get earliest date to analyze
        if period == "1m":
            start_date = datetime.now() - timedelta(days=30)
        elif period == "3m":
            start_date = datetime.now() - timedelta(days=90)
        elif period == "6m":
            start_date = datetime.now() - timedelta(days=180)
        elif period == "1y":
            start_date = datetime.now() - timedelta(days=365)
        else:
            start_date = datetime.now() - timedelta(days=365)
        
        # Get historical prices for all stocks in portfolio
        for ticker, position in self.portfolio.get("stocks", {}).items():
            shares = position.get("shares", 0)
            if shares <= 0:
                continue
                
            data = self.stock_data_service.get_stock_data(ticker, period)
            if data is not None and not data.empty:
                # Only keep dates after start_date
                data = data[data.index >= pd.Timestamp(start_date)]
                
                for date, row in data.iterrows():
                    date_str = date.strftime("%Y-%m-%d")
                    if date_str not in dates:
                        dates.append(date_str)
        
        # Sort dates
        dates.sort()
        
        # Calculate portfolio value for each date
        for date_str in dates:
            date = datetime.strptime(date_str, "%Y-%m-%d")
            value = self.portfolio.get("cash", 0)
            
            for ticker, position in self.portfolio.get("stocks", {}).items():
                shares = position.get("shares", 0)
                if shares <= 0:
                    continue
                    
                data = self.stock_data_service.get_stock_data(ticker, period)
                if data is not None and not data.empty:
                    # Find closest date
                    closest_date = min(data.index, key=lambda x: abs(x - pd.Timestamp(date)))
                    price = data.loc[closest_date, 'Close']
                    value += shares * price
            
            historical_values.append(value)
        
        # Calculate performance metrics
        if len(historical_values) > 0:
            initial_value = historical_values[0]
            final_value = historical_values[-1]
            absolute_return = final_value - initial_value
            percentage_return = (absolute_return / initial_value) * 100 if initial_value > 0 else 0
            
            # Calculate volatility
            returns = np.diff(historical_values) / historical_values[:-1]
            volatility = np.std(returns) * 100 if len(returns) > 0 else 0
            
            # Calculate Sharpe ratio (assuming risk-free rate of 2%)
            risk_free_rate = 0.02
            excess_return = (percentage_return / 100) - risk_free_rate
            sharpe_ratio = excess_return / (volatility / 100) if volatility > 0 else 0
            
            return {
                "initial_value": initial_value,
                "final_value": final_value,
                "absolute_return": absolute_return,
                "percentage_return": percentage_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "historical_values": list(zip(dates, historical_values))
            }
        else:
            return {
                "initial_value": 0,
                "final_value": current_value,
                "absolute_return": 0,
                "percentage_return": 0,
                "volatility": 0,
                "sharpe_ratio": 0,
                "historical_values": []
            }
    
    def plot_portfolio_performance(self, period="1y"):
        """
        Create a plot of portfolio performance
        
        Args:
            period (str): Time period for performance plot
            
        Returns:
            str: Path to saved plot file
        """
        performance = self.calculate_portfolio_performance(period)
        historical_values = performance.get("historical_values", [])
        
        if len(historical_values) == 0:
            return None
        
        # Extract dates and values
        dates = [date for date, _ in historical_values]
        values = [value for _, value in historical_values]
        
        # Create and save the plot
        return Visualizer.plot_portfolio_performance(dates, values, period)
    
    def add_transaction(self, transaction_type, ticker, shares, price, date=None):
        """
        Add a transaction to the portfolio
        
        Args:
            transaction_type (str): "buy" or "sell"
            ticker (str): Stock ticker symbol
            shares (float): Number of shares
            price (float): Price per share
            date (str): Transaction date (optional)
            
        Returns:
            bool: True if transaction was successful
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
            
        # Validate inputs
        if transaction_type not in ["buy", "sell"]:
            print("Invalid transaction type. Use 'buy' or 'sell'.")
            return False
            
        if shares <= 0:
            print("Shares must be positive.")
            return False
            
        if price <= 0:
            print("Price must be positive.")
            return False
        
        # Calculate transaction value
        transaction_value = shares * price
        
        # Initialize stocks dict if it doesn't exist
        if "stocks" not in self.portfolio:
            self.portfolio["stocks"] = {}
            
        # Initialize cash if it doesn't exist
        if "cash" not in self.portfolio:
            self.portfolio["cash"] = 0.0
            
        # Initialize transactions if it doesn't exist
        if "transactions" not in self.portfolio:
            self.portfolio["transactions"] = []
            
        # Process transaction
        if transaction_type == "buy":
            # Check if we have enough cash
            if transaction_value > self.portfolio["cash"]:
                print("Not enough cash for this transaction.")
                return False
                
            # Update cash
            self.portfolio["cash"] -= transaction_value
            
            # Update stock position
            if ticker not in self.portfolio["stocks"]:
                self.portfolio["stocks"][ticker] = {
                    "shares": shares,
                    "cost_basis": price
                }
            else:
                # Calculate new cost basis (weighted average)
                current_shares = self.portfolio["stocks"][ticker]["shares"]
                current_cost = self.portfolio["stocks"][ticker]["cost_basis"]
                total_shares = current_shares + shares
                new_cost_basis = ((current_shares * current_cost) + (shares * price)) / total_shares
                
                self.portfolio["stocks"][ticker]["shares"] = total_shares
                self.portfolio["stocks"][ticker]["cost_basis"] = new_cost_basis
        
        elif transaction_type == "sell":
            # Check if we have enough shares
            if ticker not in self.portfolio["stocks"] or self.portfolio["stocks"][ticker]["shares"] < shares:
                print("Not enough shares for this transaction.")
                return False
                
            # Update cash
            self.portfolio["cash"] += transaction_value
            
            # Update stock position
            remaining_shares = self.portfolio["stocks"][ticker]["shares"] - shares
            if remaining_shares <= 0:
                # Remove position if no shares left
                del self.portfolio["stocks"][ticker]
            else:
                self.portfolio["stocks"][ticker]["shares"] = remaining_shares
        
        # Record transaction
        transaction = {
            "type": transaction_type,
            "ticker": ticker,
            "shares": shares,
            "price": price,
            "value": transaction_value,
            "date": date
        }
        self.portfolio["transactions"].append(transaction)
        
        # Save updated portfolio
        self.save_portfolio()
        
        return True
    
    def normalize_ticker(self, ticker):
        """
        Normalize ticker symbols to prevent duplicates like TCS and TCS.NS
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            str: Normalized ticker symbol
        """
        # List of Indian stocks that should be normalized
        indian_stocks = Config.INDIAN_STOCKS
        
        # Remove .NS or .BO suffix if present
        base_ticker = ticker.split('.')[0]
        
        # For Indian stocks, always use the .NS suffix
        if base_ticker in indian_stocks:
            return f"{base_ticker}.NS"
        
        return ticker
    
    def add_stock(self, ticker, shares, price, date=None):
        """
        Add a stock to the portfolio (wrapper for add_transaction)
        
        Args:
            ticker (str): Stock ticker symbol
            shares (float): Number of shares
            price (float): Price per share
            date (str): Transaction date (optional)
            
        Returns:
            bool: True if transaction was successful
        """
        # Normalize the ticker to prevent duplicates
        normalized_ticker = self.normalize_ticker(ticker)
        
        # Check if we already have this stock under a different ticker format
        for existing_ticker in list(self.portfolio.get("stocks", {}).keys()):
            if self.normalize_ticker(existing_ticker) == normalized_ticker and existing_ticker != normalized_ticker:
                # We have the same stock under a different ticker format - merge them
                print(f"Merging {ticker} with existing position {existing_ticker}")
                current_shares = self.portfolio["stocks"][existing_ticker]["shares"]
                current_cost = self.portfolio["stocks"][existing_ticker]["cost_basis"]
                
                # Calculate new cost basis (weighted average)
                total_shares = current_shares + shares
                new_cost_basis = ((current_shares * current_cost) + (shares * price)) / total_shares
                
                # Update existing position
                self.portfolio["stocks"][existing_ticker]["shares"] = total_shares
                self.portfolio["stocks"][existing_ticker]["cost_basis"] = new_cost_basis
                
                # Update cash
                self.portfolio["cash"] -= shares * price
                
                # Record transaction
                transaction = {
                    "type": "buy",
                    "ticker": existing_ticker,
                    "shares": shares,
                    "price": price,
                    "value": shares * price,
                    "date": date or datetime.now().strftime("%Y-%m-%d")
                }
                self.portfolio["transactions"].append(transaction)
                
                # Save updated portfolio
                self.save_portfolio()
                
                return True
        
        # If we didn't find a match, proceed with normal add transaction
        return self.add_transaction("buy", normalized_ticker, shares, price, date)
    
    def remove_stock(self, ticker):
        """
        Remove a stock from the portfolio
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            bool: True if removal was successful
        """
        if ticker not in self.portfolio.get("stocks", {}):
            print(f"Stock {ticker} not found in portfolio.")
            return False
            
        # Remove the stock
        del self.portfolio["stocks"][ticker]
        
        # Record as a removal transaction
        transaction = {
            "type": "remove",
            "ticker": ticker,
            "date": datetime.now().strftime("%Y-%m-%d")
        }
        self.portfolio["transactions"].append(transaction)
        
        # Save updated portfolio
        self.save_portfolio()
        
        return True
    
    def calculate_portfolio_value(self):
        """
        Calculate the current total value and profit/loss of the portfolio
        
        Returns:
            dict: Portfolio value information
        """
        total_value = self.portfolio.get("cash", 0)
        cost_basis = self.portfolio.get("cash", 0)  # Start with cash
        
        for ticker, position in self.portfolio.get("stocks", {}).items():
            shares = position.get("shares", 0)
            cost_per_share = position.get("cost_basis", 0)
            
            if shares <= 0:
                continue
                
            data = self.stock_data_service.get_stock_data(ticker)
            if data is not None and not data.empty:
                current_price = data['Close'].iloc[-1]
                position_value = shares * current_price
                position_cost = shares * cost_per_share
                
                total_value += position_value
                cost_basis += position_cost
        
        # Calculate profit/loss
        profit_loss = total_value - cost_basis
        profit_loss_pct = (profit_loss / cost_basis) * 100 if cost_basis > 0 else 0
        
        return {
            "total_value": total_value,
            "cost_basis": cost_basis,
            "total_profit_loss": profit_loss,
            "total_profit_loss_pct": profit_loss_pct
        }
    
    def get_portfolio_summary(self):
        """
        Get a text summary of the portfolio
        
        Returns:
            str: Portfolio summary text
        """
        if not self.portfolio.get("stocks"):
            return "Your portfolio is empty. Add some stocks to get started!"
            
        value_info = self.calculate_portfolio_value()
        
        summary = "Portfolio Summary:\n\n"
        summary += f"Total Value: ${value_info['total_value']:.2f}\n"
        summary += f"Cost Basis: ${value_info['cost_basis']:.2f}\n"
        summary += f"Total Profit/Loss: ${value_info['total_profit_loss']:.2f} ({value_info['total_profit_loss_pct']:.2f}%)\n\n"
        
        summary += "Holdings:\n"
        summary += "-" * 60 + "\n"
        summary += f"{'Ticker':<10} {'Shares':<10} {'Cost Basis':<15} {'Current Price':<15} {'Value':<15} {'Profit/Loss':<15}\n"
        summary += "-" * 60 + "\n"
        
        # Cash position
        cash = self.portfolio.get("cash", 0)
        summary += f"{'CASH':<10} {'':<10} {'':<15} {'':<15} ${cash:<14.2f} {'':<15}\n"
        
        # Stock positions
        for ticker, position in self.portfolio.get("stocks", {}).items():
            shares = position.get("shares", 0)
            cost_basis = position.get("cost_basis", 0)
            
            data = self.stock_data_service.get_stock_data(ticker)
            if data is not None and not data.empty:
                current_price = data['Close'].iloc[-1]
                position_value = shares * current_price
                position_cost = shares * cost_basis
                profit_loss = position_value - position_cost
                profit_loss_pct = (profit_loss / position_cost) * 100 if position_cost > 0 else 0
                
                summary += f"{ticker:<10} {shares:<10.2f} ${cost_basis:<14.2f} ${current_price:<14.2f} ${position_value:<14.2f} ${profit_loss:<14.2f} ({profit_loss_pct:.2f}%)\n"
        
        summary += "-" * 60 + "\n"
        
        return summary
    
    def plot_portfolio_composition(self):
        """
        Create a pie chart of portfolio composition
        
        Returns:
            str: Path to saved plot file
        """
        # Get portfolio allocation
        allocation = self.get_portfolio_allocation()
        
        # Create and save the plot
        return Visualizer.plot_portfolio_composition(allocation)
    
    def add_cash(self, amount):
        """
        Add cash to the portfolio
        
        Args:
            amount (float): Amount of cash to add
            
        Returns:
            bool: True if successful
        """
        if amount <= 0:
            print("Amount must be positive.")
            return False
            
        # Initialize cash if it doesn't exist
        if "cash" not in self.portfolio:
            self.portfolio["cash"] = 0.0
            
        # Add cash
        self.portfolio["cash"] += amount
        
        # Record as a cash deposit transaction
        transaction = {
            "type": "deposit",
            "amount": amount,
            "date": datetime.now().strftime("%Y-%m-%d")
        }
        
        # Initialize transactions if it doesn't exist
        if "transactions" not in self.portfolio:
            self.portfolio["transactions"] = []
            
        self.portfolio["transactions"].append(transaction)
        
        # Save updated portfolio
        self.save_portfolio()
        
        return True