import numpy as np

def calculate_sharpe_ratio(returns: list[float], risk_free_rate: float) -> float:
    """
    Calculates the Sharpe ratio, a measure of risk-adjusted return.

    Args:
        returns (list[float]): A list of portfolio returns (e.g., daily, weekly).
        risk_free_rate (float): The risk-free rate of return for the same period as returns.

    Returns:
        float: The calculated Sharpe ratio.
               Returns 0.0 if the standard deviation of returns is zero and the mean excess return is also zero.
               Returns float('inf') or float('-inf') if the standard deviation is zero and the mean excess return is non-zero.

    Raises:
        ValueError: If the `returns` list is empty.
    """
    if not returns:
        raise ValueError("Returns list cannot be empty.")

    returns_array = np.array(returns)
    mean_return = np.mean(returns_array)
    std_dev_returns = np.std(returns_array)

    if std_dev_returns == 0:
        excess_return = mean_return - risk_free_rate
        if excess_return == 0:
            return 0.0
        elif excess_return > 0:
            return float('inf')
        else:
            return float('-inf')

    sharpe_ratio = (mean_return - risk_free_rate) / std_dev_returns
    return sharpe_ratio

def calculate_max_drawdown(prices: list[float]) -> float:
    """
    Calculates the maximum drawdown from a list of prices.
    Maximum drawdown is the largest peak-to-trough decline during a specific period.

    Args:
        prices (list[float]): A list of asset prices in chronological order.

    Returns:
        float: The maximum drawdown as a positive percentage (e.g., 0.1 for 10% drawdown).
               Returns 0.0 if the `prices` list is empty or contains only one element.

    Raises:
        ValueError: If the `prices` list is empty (explicitly checked, though `len(prices) <= 1` also covers it).
    """
    if not prices:
        raise ValueError("Prices list cannot be empty.")

    if len(prices) <= 1:
        return 0.0

    prices_array = np.array(prices)
    peak = prices_array[0]
    max_drawdown = 0.0

    for price in prices_array:
        if price > peak:
            peak = price
        drawdown = (peak - price) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return max_drawdown

def calculate_sortino_ratio(returns: list[float], risk_free_rate: float, target_return: float = 0.0) -> float:
    """
    Calculates the Sortino ratio, a variation of the Sharpe ratio that differentiates harmful volatility from total overall volatility
    by using the asset's standard deviation of negative asset returns—downside deviation—instead of the total standard deviation of portfolio returns.

    Args:
        returns (list[float]): A list of portfolio returns.
        risk_free_rate (float): The risk-free rate of return.
        target_return (float, optional): The minimum acceptable return, used to calculate downside deviation. Defaults to 0.0.

    Returns:
        float: The calculated Sortino ratio.
               Returns 0.0 if the downside deviation is zero and the mean excess return (over risk-free rate) is also zero.
               Returns float('inf') or float('-inf') if the downside deviation is zero and the mean excess return is non-zero.

    Raises:
        ValueError: If the `returns` list is empty.
    """
    if not returns:
        raise ValueError("Returns list cannot be empty.")

    returns_array = np.array(returns)
    mean_return = np.mean(returns_array)

    downside_returns = returns_array[returns_array < target_return]

    if not downside_returns.any(): # Handles cases where all returns are >= target_return
        downside_deviation = 0.0
    else:
        downside_deviation = np.std(downside_returns)

    if downside_deviation == 0:
        excess_return = mean_return - risk_free_rate
        if excess_return == 0: # Adjusted to compare with risk_free_rate as per common Sortino definition
            return 0.0
        elif excess_return > 0:
            return float('inf')
        else:
            return float('-inf')

    sortino_ratio = (mean_return - risk_free_rate) / downside_deviation
    return sortino_ratio
