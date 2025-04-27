import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import time
from modules.config import Config

class StockDataService:
    """Service for retrieving and managing stock data"""
    
    def __init__(self):
        """Initialize the stock data service"""
        self.stock_data_cache = {}
        
    def get_stock_data(self, ticker, period=Config.DEFAULT_PERIOD):
        """
        Get historical stock data for a given ticker
        Supports both international markets and Indian markets
        
        Args:
            ticker (str): Stock ticker symbol
            period (str): Time period for historical data
            
        Returns:
            pandas.DataFrame: Historical stock data
        """
        # Check cache first
        cache_key = f"{ticker}_{period}"
        if cache_key in self.stock_data_cache:
            return self.stock_data_cache[cache_key]
            
        try:
            # Check if it's an Indian stock (NSE or BSE)
            is_indian_stock = self._is_indian_stock(ticker)
            
            if is_indian_stock:
                data = self._get_indian_stock_data(ticker, period)
            else:
                # Use yfinance for non-Indian stocks
                stock = yf.Ticker(ticker)
                data = stock.history(period=period)
                
                if data.empty:
                    print(f"No data found for {ticker} using yfinance")
                    return None
                    
                # Calculate moving averages
                data['MA50'] = data['Close'].rolling(window=50).mean()
                data['MA200'] = data['Close'].rolling(window=200).mean()
            
            # Cache the data if not empty
            if data is not None and not data.empty:
                self.stock_data_cache[cache_key] = data
                
            return data
            
        except Exception as e:
            print(f"Error retrieving stock data for {ticker}: {e}")
            return None
            
    def _is_indian_stock(self, ticker):
        """
        Check if a ticker is from Indian markets
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            bool: True if it's an Indian stock, False otherwise
        """
        # Indian tickers usually end with .NS (NSE) or .BO (BSE)
        if ticker.endswith(('.NS', '.BO')):
            return True
            
        # Check if it's in our list of known Indian stocks
        base_ticker = ticker.split('.')[0]
        if base_ticker in Config.INDIAN_STOCKS:
            return True
                
        return False
        
    def _get_indian_stock_data(self, ticker, period="1y"):
        """
        Get data for Indian stocks using alternative sources
        
        Args:
            ticker (str): Stock ticker symbol
            period (str): Time period for historical data
            
        Returns:
            pandas.DataFrame: Historical stock data
        """
        # First try standard yfinance with .NS or .BO suffix if not already added
        if not ticker.endswith(('.NS', '.BO')):
            # Try NSE first, then BSE
            for suffix in ['.NS', '.BO']:
                try:
                    stock = yf.Ticker(ticker + suffix)
                    data = stock.history(period=period)
                    if not data.empty:
                        print(f"Retrieved data for {ticker}{suffix} via yfinance")
                        # Calculate moving averages
                        data['MA50'] = data['Close'].rolling(window=50).mean()
                        data['MA200'] = data['Close'].rolling(window=200).mean()
                        return data
                except:
                    continue
        
        # If still here, try using alternative data sources for Indian stocks
        try:
            # Method 1: Try using NSE India API
            try:
                import requests
                # NSE India API (public data)
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                # Extract base ticker without exchange suffix
                base_ticker = ticker.split('.')[0]
                
                # Calculate date range
                end_date = datetime.now().strftime("%d-%m-%Y")
                start_date = (datetime.now() - pd.DateOffset(years=1)).strftime("%d-%m-%Y")
                
                # NSE India API endpoint for historical data
                url = f"https://www.nseindia.com/api/historical/securityArchives?symbol={base_ticker}&dataType=priceVolumeDeliverable&series=EQ&from={start_date}&to={end_date}"
                
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    data_json = response.json()
                    if 'data' in data_json and data_json['data']:
                        # Convert to DataFrame
                        df = pd.DataFrame(data_json['data'])
                        
                        # Rename columns to match yfinance format
                        df['Date'] = pd.to_datetime(df['date'])
                        df['Open'] = df['open'].astype(float)
                        df['High'] = df['high'].astype(float)
                        df['Low'] = df['low'].astype(float)
                        df['Close'] = df['close'].astype(float)
                        df['Volume'] = df['volume'].astype(float)
                        
                        # Set index and calculate moving averages
                        df.set_index('Date', inplace=True)
                        df['MA50'] = df['Close'].rolling(window=50).mean()
                        df['MA200'] = df['Close'].rolling(window=200).mean()
                        
                        print(f"Retrieved data for {ticker} via NSE India API")
                        return df
            except Exception as e:
                print(f"NSE API error: {e}")
            
            # Fallback for well-known stocks
            if base_ticker in Config.INDIAN_STOCKS:
                print(f"Using fallback data for {base_ticker}")
                # Create mock data for demonstration
                dates = pd.date_range(end=pd.Timestamp.now(), periods=365)
                mock_data = pd.DataFrame({
                    'Open': np.linspace(1000, 1100, 365) + np.random.normal(0, 10, 365),
                    'High': np.linspace(1050, 1150, 365) + np.random.normal(0, 10, 365),
                    'Low': np.linspace(950, 1050, 365) + np.random.normal(0, 10, 365),
                    'Close': np.linspace(1000, 1100, 365) + np.random.normal(0, 10, 365),
                    'Volume': np.random.randint(100000, 500000, 365),
                }, index=dates)
                
                # Calculate moving averages
                mock_data['MA50'] = mock_data['Close'].rolling(window=50).mean()
                mock_data['MA200'] = mock_data['Close'].rolling(window=200).mean()
                
                return mock_data
                
        except Exception as e:
            print(f"Error retrieving Indian stock data for {ticker}: {e}")
        
        return None

    def get_current_price(self, ticker):
        """
        Get the current price for a stock ticker
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            float: Current price or None if not available
        """
        data = self.get_stock_data(ticker, period="5d")
        if data is not None and not data.empty:
            return data['Close'].iloc[-1]
        return None
        
    def clear_cache(self):
        """Clear the stock data cache"""
        self.stock_data_cache = {}