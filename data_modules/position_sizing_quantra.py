# Data manipulation
import numpy as np
import pandas as pd

# Import matplotlib and set the style
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')

# Ignore warnings
import warnings 
warnings.filterwarnings('ignore')

"""
Function to print the portfolio perfomance metrics and plot the cumulative returns
along with the drawdown.
"""


def performance_analysis(strategy_portfolio_value, benchmark_portfolio_value=None):
    # Store the final cumulative returns value
    final_returns = strategy_portfolio_value.iloc[-1]

    # Store the initial cumulative returns
    initial_returns = strategy_portfolio_value.iloc[0]

    # Store the number of trading days
    trading_days = len(strategy_portfolio_value)

    # Calculate the total returns
    total_returns = (final_returns / initial_returns - 1) * 100

    # Calculate the annualised returns
    annualised_performace = ((final_returns / initial_returns)
                             ** (252 / trading_days) - 1) * 100

    # Drawdown calculations
    # Calculate the running maximum
    running_max = np.maximum.accumulate(
        strategy_portfolio_value.dropna())

    # Ensure the value never drops below 1
    running_max[running_max < 1] = 1

    # Calculate the percentage drawdown
    running_drawdown = 100 * \
        ((strategy_portfolio_value)/running_max - 1)

    # Calculate the maximum drawdown
    max_drawdown = running_drawdown.min()

    # Calculate the return to max. drawdown ratio
    return_to_MDD_ratio = annualised_performace / max_drawdown

    # Plot the cumulative returns
    plt.figure(figsize=(15, 7))
    plt.title('Portfolio Value ($)', fontsize=14)

    plt.plot(
        strategy_portfolio_value, label="Strategy Performance")

    if benchmark_portfolio_value is not None:
        plt.plot(benchmark_portfolio_value, label="Benchmark Performance")

    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend()
    plt.show()

    # Plot max drawdown
    plt.figure(figsize=(15, 7))
    plt.title('Drawdown (%)', fontsize=14)
    plt.plot(running_drawdown, color='red')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.fill_between(running_drawdown.index,
                     running_drawdown.values, color='red')
    plt.show()

    # Print the performance metrics
    print(f"Total returns: {round(total_returns, 2)}%")
    print(f"Annualised returns (CAGR): {round(annualised_performace, 2)}%")
    print(f"Maximum drawdown (MDD): {round(max_drawdown, 2)}%")
    print(f"Return-to-MDD ratio: {abs(round(return_to_MDD_ratio, 2))}")
    
    
"""
Define the function to plot the capital used.
"""

def plot_leverage(leverage):
    # Plot the leverage ratio
    plt.figure(figsize=(15, 7))
    plt.plot(leverage, color='purple')
    plt.title('Leverage Ratio', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Leverage Ratio', fontsize=12)
    plt.show()
    

"""
Define the function to plot the capital used.
"""


def plot_portion_of_capital(portion_of_capital_used, total_portfolio_value):
    # Define the plot figure
    plt.figure(figsize=(15, 7))
    plt.plot(portion_of_capital_used, label="Capital Used", color='y')
    plt.plot(total_portfolio_value, label="Portfolio Value", color='blue')

    # Add chart titles and axis labels
    plt.title('Portion of Capital Used', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Capital ($)', fontsize=12)
    plt.legend()
    plt.show()