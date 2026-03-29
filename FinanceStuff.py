import yfinance as yf
import pandas as pd

def fetch_market_data(ticker_symbol, period="1y"):
    """
    Fetches historical market data for a given ticker.
    """
    print(f"Pulling data for {ticker_symbol}...")
    asset = yf.Ticker(ticker_symbol)
    
    # Fetch historical data
    df = asset.history(period=period)
    
    # Clean up the dataframe to keep only what we need for ML/Dashboards
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.reset_index(inplace=True)
    
    # Ensure the Date column is timezone-naive for easier processing later
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    
    return df

# 1. Fetch Local Index Fund (PSE)
fmetf_data = fetch_market_data("FMETF.PS")

# 2. Fetch Global Crypto
btc_data = fetch_market_data("BTC-USD")

# Display the most recent data
print("\n--- FMETF (Local Index) - Last 3 Days ---")
print(fmetf_data.tail(3))

print("\n--- BTC-USD (Global Crypto) - Last 3 Days ---")
print(btc_data.tail(3))