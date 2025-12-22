import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting

# Step 1: Choose assets (stocks)
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]

# Step 2: Download historical price data
prices = yf.download(tickers, start="2019-01-01", end="2024-01-01", auto_adjust=False)["Adj Close"]

# Step 3: Calculate expected returns (mean return)
mu = expected_returns.mean_historical_return(prices)

# Step 4: Calculate risk (covariance matrix)
Sigma = risk_models.sample_cov(prices)

# Step 5: Create the optimizer object
ef = EfficientFrontier(mu, Sigma)

# Step 6: Optimize for maximum Sharpe Ratio
weights = ef.max_sharpe()

# Step 7: Clean the weights (remove tiny values)
cleaned_weights = ef.clean_weights()

# Step 8: Print optimal weights
print("Optimal Portfolio Weights:")
print(cleaned_weights)

# Step 9: Get portfolio performance
expected_return, volatility, sharpe_ratio = ef.portfolio_performance()

print("\nPortfolio Performance:")
print(f"Expected Annual Return: {expected_return:.2%}")
print(f"Annual Volatility (Risk): {volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")


