import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Function to calculate intrinsic value using the Graham formula
def graham_intrinsic_value(eps, growth_rate, dividend_yield, risk_free_rate):
    return (eps * (8.5 + 2 * growth_rate) * 4.4) / risk_free_rate

# Function to calculate the margin of safety
def margin_of_safety(current_price, intrinsic_value):
    return (intrinsic_value - current_price) / current_price

# Function to calculate financial ratios
def calculate_ratios(data):
    data['P/E'] = data['Close'] / data['EPS'].replace(0, np.nan)
    data['P/B'] = data['Close'] / data['Book Value Per Share'].replace(0, np.nan)
    data['Dividend Yield'] = data['Dividends'].replace(0, np.nan) / data['Close'].replace(0, np.nan)
    data['ROE'] = data['Net Income'].replace(0, np.nan) / data['Shareholder Equity'].replace(0, np.nan)
    data['Debt-to-Equity'] = data['Total Liabilities'].replace(0, np.nan) / data['Shareholder Equity'].replace(0, np.nan)
    data['Current Ratio'] = data['Current Assets'].replace(0, np.nan) / data['Current Liabilities'].replace(0, np.nan)
    data['Asset Turnover'] = data['Revenue'].replace(0, np.nan) / data['Total Assets'].replace(0, np.nan)
    return data

# Function to perform fundamental analysis using machine learning
def fundamental_analysis(data):
    # Prepare the features and target variable
    features = ['P/E', 'P/B', 'Dividend Yield', 'ROE', 'Debt-to-Equity', 'Current Ratio', 'Asset Turnover']
    target = 'Close'

    # Create a pipeline for data preprocessing and model training
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(random_state=42))
    ])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

    # Perform grid search for hyperparameter tuning
    param_grid = {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [None, 5, 10],
        'rf__min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    # Make predictions on the testing set
    y_pred = best_model.predict(X_test)

    # Evaluate the model's performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    # Use the trained model to make predictions on the entire dataset
    data['Predicted Price'] = best_model.predict(data[features])

    return data

# Function to optimize portfolio weights using the Sharpe ratio
def optimize_portfolio(portfolio, stock_data):
    # Merge the portfolio and stock data
    merged_data = pd.merge(portfolio, stock_data, on='Stock')

    # Calculate the portfolio returns
    merged_data['Portfolio Return'] = merged_data['Weight'] * merged_data['Close'].pct_change()

    # Define the objective function to minimize (negative Sharpe ratio)
    def objective(weights, returns, risk_free_rate):
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
        return -sharpe_ratio

    # Define the constraints for portfolio weights
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(portfolio)))

    # Initial guess for portfolio weights
    initial_guess = np.array(portfolio['Weight'])

    # Perform optimization using minimize function from scipy.optimize
    results = minimize(objective, initial_guess, args=(merged_data['Portfolio Return'], 0.03),
                       method='SLSQP', bounds=bounds, constraints=constraints)

    # Retrieve the optimized weights
    optimized_weights = results.x

    # Update the portfolio weights with the optimized weights
    portfolio['Weight'] = optimized_weights

    return portfolio

# Function to calculate value at risk (VaR) and conditional value at risk (CVaR)
def calculate_risk_metrics(portfolio_returns, confidence_level=0.95):
    sorted_returns = np.sort(portfolio_returns)
    index = int(len(sorted_returns) * (1 - confidence_level))
    var = sorted_returns[index]
    cvar = np.mean(sorted_returns[:index])
    return var, cvar

# Function to backtest the investment strategy
def backtest_strategy(portfolio, stock_data):
    # Merge the portfolio and stock data
    merged_data = pd.merge(portfolio, stock_data, on='Stock')

    # Calculate the portfolio returns
    merged_data['Portfolio Return'] = merged_data['Weight'] * merged_data['Close'].pct_change()
    portfolio_return = merged_data['Portfolio Return'].sum()

    # Calculate the portfolio risk (standard deviation)
    portfolio_risk = merged_data['Portfolio Return'].std()

    # Calculate the Sharpe ratio
    risk_free_rate = 0.03
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk

    # Calculate value at risk (VaR) and conditional value at risk (CVaR)
    var, cvar = calculate_risk_metrics(merged_data['Portfolio Return'])

    print(f"Portfolio Return: {portfolio_return}")
    print(f"Portfolio Risk: {portfolio_risk}")
    print(f"Sharpe Ratio: {sharpe_ratio}")
    print(f"Value at Risk (VaR): {var}")
    print(f"Conditional Value at Risk (CVaR): {cvar}")

    return portfolio

# Download historical stock data
def download_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Main function
def main():
    # Define the list of stocks to analyze
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']

    # Set the start and end dates for historical data
    start_date = '2010-01-01'
    end_date = '2023-06-08'

    # Set the risk-free rate and other parameters
    risk_free_rate = 0.03
    margin_of_safety_threshold = 0.5

    # Create an empty list to store the selected stocks
    selected_stocks = []

    # Create an empty dictionary to store the stock data
    stock_data_dict = {}

    # Iterate over the stocks
    for stock in stocks:
        try:
            # Download historical stock data
            stock_data = download_stock_data(stock, start_date, end_date)

            # Calculate financial ratios
            stock_data = calculate_ratios(stock_data)

            # Perform fundamental analysis using machine learning
            stock_data = fundamental_analysis(stock_data)

            # Store the stock data in the dictionary
            stock_data_dict[stock] = stock_data

            # Calculate the intrinsic value using the most recent data
            eps = stock_data['EPS'].iloc[-1]
            growth_rate = 0.1  # Assume a constant growth rate of 10%
            dividend_yield = stock_data['Dividend Yield'].iloc[-1]
            intrinsic_value = graham_intrinsic_value(eps, growth_rate, dividend_yield, risk_free_rate)

            # Calculate the margin of safety
            current_price = stock_data['Close'].iloc[-1]
            margin_of_safety = margin_of_safety(current_price, intrinsic_value)

            # Check if the stock meets the margin of safety criteria
            if margin_of_safety >= margin_of_safety_threshold:
                selected_stocks.append(stock)

        except Exception as e:
            print(f"Error analyzing {stock}: {str(e)}")
            continue

    # Construct the portfolio with equal weightage
    portfolio = pd.DataFrame({'Stock': selected_stocks, 'Weight': [1/len(selected_stocks)] * len(selected_stocks)})

    # Optimize the portfolio weights using the Sharpe ratio
    optimized_portfolio = optimize_portfolio(portfolio, pd.concat(stock_data_dict.values()))

    # Backtest the investment strategy for the optimized portfolio
    print("\nBacktesting the Optimized Portfolio:")
    backtest_strategy(optimized_portfolio, pd.concat(stock_data_dict.values()))

    # Print the optimized portfolio weights
    print("\nOptimized Portfolio Weights:")
    print(optimized_portfolio)

# Run the main function
if __name__ == '__main__':
    main()