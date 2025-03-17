import yfinance as yf
import numpy as np

class StockParameters:
    """
    A class to fetch and compute stock parameters from historical data.
    """

    def __init__(self, ticker: str, lookback_years: int = 1):
        """
        Initialize the StockParameters instance.

        :param ticker: Stock ticker symbol (e.g., 'AAPL').
        :param lookback_years: Number of years to look back for historical data.
        """
        self.ticker = ticker
        self.lookback_years = lookback_years
        self.initial_price = None
        self.volatility = None
        self.drift = None

    def fetch_parameters(self) -> dict:
        """
        Downloads historical stock data, computes and stores the initial price, 
        annualized volatility, and annualized drift.

        :return: A dictionary with keys 'initial_price', 'volatility', and 'drift'.
        """
        # Download historical data for the given ticker
        stock = yf.Ticker(self.ticker)
        hist = stock.history(period=f"{self.lookback_years}y")
        
        if hist.empty:
            raise ValueError(f"No historical data found for ticker: {self.ticker}")

        # Calculate daily returns using the natural logarithm
        hist['return'] = np.log(hist['Close'] / hist['Close'].shift(1))
        hist = hist.dropna()

        # Compute daily volatility and drift
        daily_volatility = hist['return'].std()
        daily_drift = hist['return'].mean()

        # Annualize volatility and drift (assuming 252 trading days)
        self.volatility = daily_volatility * np.sqrt(252)
        self.drift = daily_drift * 252

        # Get current price (latest closing price)
        self.initial_price = hist['Close'].iloc[-1]

        return {
            'initial_price': self.initial_price,
            'volatility': self.volatility,
            'drift': self.drift
        }