import pandas as pd
import yfinance as yf

# Download historical data for BTC/USD
data = yf.download('BTC-USD', interval='1h', start='2024-01-01', end='2025-01-01')

# Save the data to a Parquet file
data.to_parquet('BTCUSD3600.pq')