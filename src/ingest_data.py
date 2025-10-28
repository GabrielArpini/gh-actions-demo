import yfinance as yf 
import sqlite3 
import pickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os 
import pandas as pd
from dataclasses import dataclass
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
warnings.filterwarnings('ignore')

@dataclass
class Config:
    """Configuration for stock data ingestion and forecasting"""
    stocks: list = None
    yahoo_period: str = '1y'  # Period to download from Yahoo (1d, 5d, 1mo, 3mo, 6mo, 1y, etc)
    forecast_days: int = 30     # Number of days to forecast into future
    historical_days: int = 90   # Days of historical data to show in plot
    
    def __post_init__(self):
        if self.stocks is None:
            self.stocks = ["MSFT", "AAPL", "GOOG", "TSLA", "AMZN"]

# Initialize config
config = Config()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
PLOT_DIR = os.path.join(ROOT_DIR, 'website', 'plots')
DB_PATH = os.path.join(DATA_DIR, 'predictions.db')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

def get_cursor():
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        return cur 
    except Exception as e:
        print(f"Erro connectin to db: {e}")

def make_plot(symbol, historical, forecast_df):
    """
    Create plot with historical data and forecast
    """
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(historical.index, historical, label="Historical", color="blue", linewidth=2)
    
    # Plot forecast
    plt.plot(forecast_df['ds'], forecast_df['yhat'], label="Forecast", color="red", linestyle="--", linewidth=2)
    
    # Plot confidence interval
    plt.fill_between(
        forecast_df['ds'],
        forecast_df['yhat_lower'],
        forecast_df['yhat_upper'],
        color='red', alpha=0.2, label="95% CI"
    )
    
    # Get current date and last forecast date for title
    current_date = datetime.now().strftime('%Y-%m-%d')
    last_forecast_date = forecast_df['ds'].iloc[-1].strftime('%Y-%m-%d')
    
    plt.title(f"{symbol} - ARIMA {config.forecast_days}-Day Forecast (Updated: {current_date}, Through: {last_forecast_date})", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price ($)", fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(PLOT_DIR, f'{symbol}.png')
    print(f"Saving plot to: {plot_path}")  
    plt.savefig(plot_path, dpi=100)
    plt.close()
    print(f"Plot saved: {os.path.exists(plot_path)}")


def forecast_with_arima(close_prices, forecast_days=30):
    """
    Auto-selects best parameters (p, d, q) using AIC.
    """
    print("  Fitting ARIMA model (this may take a moment)...")
    
    # Use auto_arima to find best parameters
    model = auto_arima(
        close_prices,
        start_p=1, start_q=1,
        max_p=5, max_q=5,
        d=None,  # Let it auto-determine differencing
        seasonal=False,
        stepwise=True,  # Faster
        suppress_warnings=True,
        error_action='ignore',
        trace=False
    )
    
    print(f"  Best ARIMA order: {model.order}")
    
    # Forecast future values
    forecast_result = model.predict(n_periods=forecast_days, return_conf_int=True)
    forecast_values = forecast_result[0]
    conf_int = forecast_result[1]
    
    # Generate future business dates
    last_date = close_prices.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_days)
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': forecast_values,
        'yhat_lower': conf_int[:, 0],
        'yhat_upper': conf_int[:, 1]
    })
    
    return forecast_df, model


def download_stocks():
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        symbol TEXT,
        date TEXT,
        current_price REAL,
        forecast_price REAL,
        percent_change REAL,
        signal TEXT,
        signal_strength REAL,
        predicted_high REAL,
        predicted_low REAL,
        volatility REAL,
        trained_at TEXT
    )
    ''')

    for symbol in config.stocks:
        print(f"\nDownloading {symbol}...")
        symbol_data = yf.download(symbol, period=config.yahoo_period, progress=False)
        
        # Fix: Flatten multi-level columns if they exist
        if isinstance(symbol_data.columns, pd.MultiIndex):
            symbol_data.columns = symbol_data.columns.get_level_values(0)
        
        if symbol_data.empty or 'Close' not in symbol_data.columns:
            print(f"Warning: No data for {symbol}. Skipping.")
            continue
        
        close = symbol_data['Close']
        
        # Ensure close is a Series (flatten if needed)
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        try:
            forecast_df, model = forecast_with_arima(close, config.forecast_days)
        except Exception as e:
            print(f"  Error fitting ARIMA for {symbol}: {e}")
            print(f"  Skipping {symbol}")
            continue
        
        # Get historical data for plot (controlled by config.historical_days)
        historical = close[-config.historical_days:]
        
        # Calculate metrics and signals
        current_price = close.iloc[-1].item() if hasattr(close.iloc[-1], 'item') else float(close.iloc[-1])
        next_day_forecast = forecast_df['yhat'].iloc[0]
        percent_change = ((next_day_forecast - current_price) / current_price) * 100
        
        # Signal strength based on percent change
        signal_strength = abs(percent_change)
        if percent_change > 2:
            signal = "bullish"
        elif percent_change < -2:
            signal = "bearish"
        else:
            signal = "neutral"
        
        # Predicted range
        predicted_high = forecast_df['yhat_upper'].max()
        predicted_low = forecast_df['yhat_lower'].min()
        
        # Volatility (average confidence interval width)
        volatility = (forecast_df['yhat_upper'] - forecast_df['yhat_lower']).mean()
        
        # Save model
        model_path = os.path.join(DATA_DIR, f'{symbol}_latest.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Insert into database
        cur.execute('''
        INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol, 
            datetime.now().isoformat(), 
            float(current_price), 
            float(next_day_forecast),
            float(percent_change),
            signal,
            float(signal_strength),
            float(predicted_high),
            float(predicted_low),
            float(volatility),
            datetime.now().isoformat()
        ))
        
        print(f"  {symbol}: Current=${current_price:.2f}, Forecast=${next_day_forecast:.2f} ({percent_change:+.2f}%), Signal={signal}")
        
        # Generate plot
        make_plot(symbol, historical, forecast_df)
    
    conn.commit()
    conn.close()
    print("\n All stocks processed successfully!")

if __name__ == '__main__':
    download_stocks()
