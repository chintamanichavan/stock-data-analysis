import yfinance as yf

# data = yf.download("AAPL", start="2020-01-01", end="2024-03-16")
# print(data.head())

# apple = yf.Ticker("AAPL")
# print(apple.info)  # General information about Apple Inc.

# recent_data = yf.download("AAPL", period="5d")
# print(recent_data)

# multi_data = yf.download(["AAPL", "MSFT"], start="2020-01-01", end="2021-01-01")
# print(multi_data)

# data = yf.download("AAPL", start="2020-01-01", end="2021-01-01", auto_adjust=True)
# print(data['Close'])  # This will show the adjusted close prices

# weekly_data = yf.download("AAPL", start="2020-01-01", end="2021-01-01", interval="1wk")
# print(weekly_data)

# apple = yf.Ticker("AAPL")
# dividends = apple.dividends
# splits = apple.splits
# print(dividends, splits)

# apple = yf.Ticker("AAPL")
# print(apple.history(period="1d"))

# data = yf.download("AAPL", start="2022-01-01", end="2022-12-31")
# print(data)

import matplotlib.pyplot as plt

data = yf.download("AAPL", start="2020-01-01", end="2021-01-01")
data['Close'].plot()
plt.title("Apple Stock Prices")
plt.show()
