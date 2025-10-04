# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')

# Ignore warnings
import warnings 
warnings.filterwarnings('ignore')

def run_vol_targeting(spy_dataframe, volatility_target, window, leverage_cap, initial_capital):
    # Create a copy of the dataframe to avoid changing the original dataframe
    spy_df = spy_dataframe.copy()

    '''Calculate benchmark strategy returns'''
    #######################################################################
    spy_df['returns'] = spy_df['Close'].pct_change()
    spy_df.dropna(inplace=True)

    '''Calculate the volatility adjusted leverage'''
    #######################################################################
    # Calculate the volatility of the asset returns
    spy_df['volatility'] = spy_df['returns'].rolling(window).std()
    spy_df.dropna(inplace=True)

    # Calculate the leverage based on target volatility
    spy_df['leverage'] = volatility_target / spy_df['volatility'].shift(1)

    # Cap the leverage to the maximum value
    spy_df['leverage'] = np.where(
        spy_df['leverage'] > leverage_cap, leverage_cap, spy_df['leverage'])
    spy_df.dropna(inplace=True)

    '''Calculate levered returns'''
    #######################################################################
    # Calculate the levered return for that day
    spy_df['levered_return'] = spy_df['leverage'] * \
        spy_df['returns']

    # Calculate the final account value from the position sizing strategy
    spy_df['strategy_account_value'] = initial_capital * \
        (1 + spy_df['levered_return']).cumprod()

    # Calculate the final account value from the benchmark strategy
    spy_df['benchmark_account_value'] = initial_capital * \
        (1 + spy_df['returns']).cumprod()

    return spy_df


def run_cppi(spy_dataframe, m, floor_percent, window, initial_capital):
    # Create a copy of the dataframe to avoid changing the original dataframe
    spy_df = spy_dataframe.copy()

    '''Calculate benchmark strategy returns'''
    #######################################################################
    spy_df['returns'] = spy_df['Close'].pct_change()
    spy_df.dropna(inplace=True)

    # Define empty columns for various values
    spy_df['leverage'] = np.nan
    spy_df['levered_return'] = np.nan
    spy_df['strategy_account_value'] = initial_capital
    spy_df['benchmark_account_value'] = initial_capital

    '''Set up paramaeters for CPPI'''
    #######################################################################
    cushion_percentage = 1 - floor_percent

    # Absolute values
    floor = floor_percent * initial_capital
    cushion = cushion_percentage * initial_capital

    # Intialise variables
    account_value = initial_capital

    # Run CPPI
    for row in range(len(spy_df)):
        # The risky asset returns will be multiplied by `m`
        levered_return = m * spy_df['returns'].iloc[row]

        # Update account value and append to DF
        account_value = floor + (cushion * (1 + levered_return))

        # Recalculate cushion
        cushion = account_value - floor

        # Update leverage and append to DF
        leverage = m * (cushion / account_value)
        spy_df['leverage'].iloc[row] = leverage

        # Calculate the levered return for that day
        spy_df['levered_return'].iloc[row] = leverage * \
            spy_df['returns'].iloc[row]

    # Calculate the final account value from the position sizing strategy
    spy_df['strategy_account_value'] = initial_capital * \
        (1 + spy_df['levered_return']).cumprod()

    # Calculate the final account value from the benchmark strategy
    spy_df['benchmark_account_value'] = initial_capital * \
        (1 + spy_df['returns']).cumprod()

    return spy_df


def run_tipp(spy_dataframe, m, floor_percent, window, initial_capital):
    # Create a copy of the dataframe to avoid changing the original dataframe
    spy_df = spy_dataframe.copy()

    '''Calculate benchmark strategy returns'''
    #######################################################################
    spy_df['returns'] = spy_df['Close'].pct_change()
    spy_df.dropna(inplace=True)

    # Define empty columns for various values
    spy_df['leverage'] = np.nan
    spy_df['levered_return'] = np.nan
    spy_df['strategy_account_value'] = initial_capital
    spy_df['benchmark_account_value'] = initial_capital

    '''Set up paramaeters for TIPP'''
    #######################################################################
    cushion_percentage = 1 - floor_percent

    # Absolute values
    floor = floor_percent * initial_capital
    cushion = cushion_percentage * initial_capital

    # Intialise variables
    account_value = initial_capital
    max_account_value = initial_capital

    # Run TIPP
    for row in range(len(spy_df)):
        # The risky asset returns will be multiplied by `m`
        levered_return = m * spy_df['returns'].iloc[row]

        # Update account value and append to DF
        account_value = floor + (cushion * (1 + levered_return))

        # Check if account_value exceeds max_account_value
        if (account_value > max_account_value):
            # If current account value > max account value, recalculate floor
            floor = floor_percent * account_value

            # Update max_account_value
            max_account_value = account_value

        # Recalculate cushion
        cushion = account_value - floor

        # Update leverage and append to DF
        leverage = m * (cushion / account_value)
        spy_df['leverage'].iloc[row] = leverage

        # Calculate the levered return for that day
        spy_df['levered_return'].iloc[row] = leverage * \
            spy_df['returns'].iloc[row]

    # Calculate the final account value from the position sizing strategy
    spy_df['strategy_account_value'] = initial_capital * \
        (1 + spy_df['levered_return']).cumprod()

    # Calculate the final account value from the benchmark strategy
    spy_df['benchmark_account_value'] = initial_capital * \
        (1 + spy_df['returns']).cumprod()

    return spy_df


def run_tipp_vol(spy_dataframe, m, floor_percent, volatility_target, window, leverage_cap, initial_capital):
    # Create a copy of the dataframe to avoid changing the original dataframe
    spy_df = spy_dataframe.copy()

    '''Calculate benchmark strategy returns'''
    #######################################################################
    spy_df['returns'] = spy_df['Close'].pct_change()
    spy_df.dropna(inplace=True)

    '''Calculate the volatility adjusted leverage'''
    #######################################################################
    # Calculate the volatility of the asset returns
    spy_df['volatility'] = spy_df['returns'].rolling(window).std()
    spy_df.dropna(inplace=True)

    # Calculate the leverage based on target volatility
    spy_df['leverage'] = volatility_target / spy_df['volatility'].shift(1)

    # Cap the leverage to the maximum value
    spy_df['leverage'] = np.where(
        spy_df['leverage'] > leverage_cap, leverage_cap, spy_df['leverage'])
    spy_df.dropna(inplace=True)

    # Define empty columns for various values
    spy_df['levered_return'] = np.nan
    spy_df['strategy_account_value'] = initial_capital
    spy_df['benchmark_account_value'] = initial_capital

    '''Set up paramaeters for TIPP with volatility targeting'''
    #######################################################################
    cushion_percentage = 1 - floor_percent

    # Absolute values
    floor = floor_percent * initial_capital
    cushion = cushion_percentage * initial_capital

    # Intialise variables
    account_value = initial_capital
    max_account_value = initial_capital

    # Run TIPP with volatility targeting
    for row in range(len(spy_df)):
        # Adjust the multiplier w.r.t. to the leverage based on volatility
        adj_multiplier = m * spy_df['leverage'].iloc[row]

        # The risky asset returns will be multiplied by `m`
        levered_return = adj_multiplier * spy_df['returns'].iloc[row]

        # Update account value and append to DF
        account_value = floor + (cushion * (1 + levered_return))

        # Check if account_value exceeds max_account_value
        if (account_value > max_account_value):
            # If current account value > max account value, recalculate floor
            floor = floor_percent * account_value

            # Update max_account_value
            max_account_value = account_value

        # Recalculate cushion
        cushion = account_value - floor

        # Update leverage and append to DF
        leverage = adj_multiplier * (cushion / account_value)
        spy_df['leverage'].iloc[row] = leverage

        # Calculate the levered return for that day
        spy_df['levered_return'].iloc[row] = leverage * \
            spy_df['returns'].iloc[row]

    # Calculate the final account value from the position sizing strategy
    spy_df['strategy_account_value'] = initial_capital * \
        (1 + spy_df['levered_return']).cumprod()

    # Calculate the final account value from the benchmark strategy
    spy_df['benchmark_account_value'] = initial_capital * \
        (1 + spy_df['returns']).cumprod()

    return spy_df


def performance_analysis(strategy_dataframe, initial_capital):
    strategy_portfolio_value = strategy_dataframe['strategy_account_value']
    benchmark_portfolio_value = strategy_dataframe['benchmark_account_value']

    # Store the final cumulative returns value
    final_returns = strategy_portfolio_value.iloc[-1]

    # Store the number of trading days
    trading_days = len(strategy_portfolio_value)

    # Calculate the total returns
    total_returns = (final_returns / initial_capital - 1) * 100

    # Calculate the annualised returns
    annualised_performace = ((final_returns / initial_capital)
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

    # Sharpe Ratio
    sharpe_ratio = strategy_dataframe['levered_return'].mean(
    ) / strategy_dataframe['levered_return'].std()

    # Plot the cumulative returns
    plt.figure(figsize=(15, 7))
    plt.title('Portfolio Value ($)', fontsize=14)

    plt.plot(
        strategy_portfolio_value, label="Strategy Performance")

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
    print(f"Sharpe ratio: {(round(sharpe_ratio, 2))}")