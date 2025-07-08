# Calculating Beta and Expected Return of TCS.NS  ITC.NS  WIPRO.NS using CAPM
# Install required libraries if not already installed
import sys
import subprocess

def install_if_needed(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in ['yfinance', 'pandas', 'numpy', 'matplotlib', 'scikit-learn']:
    install_if_needed(pkg)

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Define stocks and index
stock_tickers = ['TCS.NS', 'ITC.NS', 'WIPRO.NS']
index_ticker = '^NSEI'
all_tickers = stock_tickers + [index_ticker]

# Download 1 year of daily adjusted close prices
data = yf.download(all_tickers, period='1y', interval='1d')['Close'].dropna()


# Calculate daily returns
returns = data.pct_change().dropna()
stock_returns = returns[stock_tickers]
index_returns = returns[index_ticker]

# Risk-free rate and market return (as decimals)
risk_free_rate = 0.065
market_return = 0.12

# Prepare results storage
results = []

# Analysis and regression visualization
for stock in stock_tickers:
    X = index_returns.values.reshape(-1, 1)
    y = stock_returns[stock].values
    model = LinearRegression()
    model.fit(X, y)
    beta = model.coef_[0]
    intercept = model.intercept_
    capm_return = risk_free_rate + beta * (market_return - risk_free_rate)
    actual_return = stock_returns[stock].mean() * 252  # Annualized
    
    results.append({
        'Stock': stock,
        'Beta': beta,
        'Expected Return (CAPM)': capm_return,
        'Actual Return (Annualized)': actual_return
    })

    # Regression plot
    plt.figure(figsize=(7, 5))
    plt.scatter(X, y, alpha=0.3, label='Daily Returns')
    plt.plot(X, model.predict(X), color='red', label=f'Regression Line (Beta={beta:.2f})')
    plt.title(f'{stock} Returns vs Nifty 50 Returns')
    plt.xlabel('Nifty 50 Daily Returns')
    plt.ylabel(f'{stock} Daily Returns')
    plt.legend()
    plt.grid(True)
    plt.show()

# Display results in a table
results_df = pd.DataFrame(results)
results_df['Beta'] = results_df['Beta'].round(3)
results_df['Expected Return (CAPM)'] = (results_df['Expected Return (CAPM)'] * 100).round(2).astype(str) + '%'
results_df['Actual Return (Annualized)'] = (results_df['Actual Return (Annualized)'] * 100).round(2).astype(str) + '%'

print("\nBeta and Expected Return (CAPM) vs Actual Return for Each Stock:\n")
print(results_df.to_string(index=False))

