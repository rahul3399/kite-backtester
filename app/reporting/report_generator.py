def _format_metric_name(metric_key: str) -> str:
    """
    Converts a metric key string (e.g., 'sharpe_ratio') into a more readable
    title-cased string (e.g., 'Sharpe Ratio').

    Args:
        metric_key (str): The metric key to format.

    Returns:
        str: The formatted, human-readable metric name.
    """
    return ' '.join(word.capitalize() for word in metric_key.split('_'))

def generate_text_report(metrics: dict[str, float], strategy_name: str) -> str:
    """
    Generates a simple text-based report summarizing a trading strategy's performance metrics.

    The report lists each metric name and its value. 'max_drawdown' is presented as a percentage.
    Other float values are formatted to two decimal places. If the metrics dictionary is empty,
    a message indicating no available metrics is returned.

    Args:
        metrics (dict[str, float]): A dictionary where keys are metric names (e.g., 'sharpe_ratio')
                                     and values are the corresponding float values.
        strategy_name (str): The name of the trading strategy.

    Returns:
        str: A string formatted as a multi-line text report.
             Example:
             Strategy Performance Report: MyStrategy
             -------------------------------------------
             Sharpe Ratio: 1.50
             Maximum Drawdown: 10.00%
             -------------------------------------------
    """
    if not metrics:
        return f"No metrics available for strategy: {strategy_name}"

    report_lines = [
        f"Strategy Performance Report: {strategy_name}",
        "-------------------------------------------"
    ]

    for key, value in metrics.items():
        readable_name = _format_metric_name(key)
        if key == 'max_drawdown':
            report_lines.append(f"{readable_name}: {value * 100:.2f}%")
        else:
            report_lines.append(f"{readable_name}: {value:.2f}")

    report_lines.append("-------------------------------------------")
    return "\n".join(report_lines)

def generate_html_report(metrics: dict[str, float], strategy_name: str) -> str:
    """
    Generates a basic HTML report summarizing a trading strategy's performance metrics.

    The report displays the metrics in an HTML table. 'max_drawdown' is presented as a percentage.
    Other float values are formatted to two decimal places. If the metrics dictionary is empty,
    a paragraph indicating no available metrics is returned.

    Args:
        metrics (dict[str, float]): A dictionary where keys are metric names (e.g., 'sharpe_ratio')
                                     and values are the corresponding float values.
        strategy_name (str): The name of the trading strategy.

    Returns:
        str: A string containing the HTML report.
             Includes a title, header, and a table for metrics.
             Example table row: <tr><td>Sharpe Ratio</td><td>1.50</td></tr>
    """
    if not metrics:
        return f"<p>No metrics available for strategy: {strategy_name}</p>"

    html_rows = []
    for key, value in metrics.items():
        readable_name = _format_metric_name(key)
        if key == 'max_drawdown':
            formatted_value = f"{value * 100:.2f}%"
        else:
            formatted_value = f"{value:.2f}"
        html_rows.append(f"  <tr><td>{readable_name}</td><td>{formatted_value}</td></tr>")

    html_table = "\n".join(html_rows)

    return f"""<!DOCTYPE html>
<html>
<head>
<title>Strategy Performance Report: {strategy_name}</title>
<style>
  table {{ font-family: arial, sans-serif; border-collapse: collapse; width: 50%; margin-left: auto; margin-right: auto; }}
  td, th {{ border: 1px solid #dddddd; text-align: left; padding: 8px; }}
  tr:nth-child(even) {{ background-color: #f2f2f2; }}
  h2 {{ text-align: center; }}
</style>
</head>
<body>
<h2>Strategy Performance Report: {strategy_name}</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
{html_table}
</table>
</body>
</html>
"""
