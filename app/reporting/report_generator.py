# app/reporting/report_generator.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import io
from jinja2 import Template

from .metrics_calculator import MetricsCalculator
from ..backtesting.engine import BacktestResult

logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Comprehensive report generator for backtesting and paper trading results
    Generates detailed reports in multiple formats (JSON, CSV, Excel, HTML, PDF)
    """
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize report generator
        
        Args:
            template_dir: Directory containing report templates
        """
        self.metrics_calculator = MetricsCalculator()
        self.template_dir = Path(template_dir) if template_dir else Path("app/reporting/templates")
        
        # Report sections configuration
        self.report_sections = [
            'summary',
            'performance_metrics',
            'risk_analysis',
            'trade_analysis',
            'position_analysis',
            'time_analysis',
            'drawdown_analysis',
            'monthly_analysis',
            'symbol_analysis'
        ]
        
    def generate_backtest_report(self, 
                               result: BacktestResult,
                               format: str = "json",
                               include_trades: bool = True,
                               include_equity_curve: bool = True,
                               output_path: Optional[str] = None) -> Union[Dict[str, Any], str, bytes]:
        """
        Generate comprehensive backtest report
        
        Args:
            result: BacktestResult object
            format: Output format (json, csv, excel, html, pdf)
            include_trades: Include detailed trade list
            include_equity_curve: Include equity curve data
            output_path: Optional path to save report
            
        Returns:
            Report in requested format
        """
        
        # Build comprehensive report data
        report_data = self._build_report_data(result, include_trades, include_equity_curve)
        
        # Generate report in requested format
        if format == "json":
            report = self._generate_json_report(report_data)
        elif format == "csv":
            report = self._generate_csv_report(report_data)
        elif format == "excel":
            report = self._generate_excel_report(report_data)
        elif format == "html":
            report = self._generate_html_report(report_data)
        elif format == "pdf":
            report = self._generate_pdf_report(report_data)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        # Save if output path provided
        if output_path:
            self._save_report(report, output_path, format)
            
        return report
        
    def generate_comparison_report(self,
                                 results: List[BacktestResult],
                                 format: str = "json") -> Union[Dict[str, Any], str, bytes]:
        """
        Generate comparison report for multiple backtests
        
        Args:
            results: List of BacktestResult objects
            format: Output format
            
        Returns:
            Comparison report
        """
        
        comparison_data = {
            'comparison_date': datetime.now().isoformat(),
            'total_backtests': len(results),
            'strategies': []
        }
        
        # Extract key metrics for each result
        for result in results:
            strategy_data = {
                'name': result.strategy_name,
                'symbols': result.symbols,
                'period': {
                    'start': result.start_date.isoformat(),
                    'end': result.end_date.isoformat()
                },
                'metrics': result.metrics,
                'summary': {
                    'total_return': result.metrics.get('total_return', 0),
                    'sharpe_ratio': result.metrics.get('sharpe_ratio', 0),
                    'max_drawdown': result.metrics.get('max_drawdown', 0),
                    'win_rate': result.metrics.get('win_rate', 0),
                    'profit_factor': result.metrics.get('profit_factor', 0),
                    'total_trades': len(result.trades)
                }
            }
            comparison_data['strategies'].append(strategy_data)
            
        # Rank strategies by different metrics
        comparison_data['rankings'] = self._rank_strategies(comparison_data['strategies'])
        
        # Generate statistical comparison
        comparison_data['statistics'] = self._compare_statistics(comparison_data['strategies'])
        
        if format == "json":
            return comparison_data
        elif format == "html":
            return self._generate_comparison_html(comparison_data)
        else:
            return comparison_data
            
    def generate_optimization_report(self,
                                   optimization_results: List[Dict[str, Any]],
                                   parameter_grid: Dict[str, List[Any]],
                                   optimization_metric: str) -> Dict[str, Any]:
        """
        Generate optimization report
        
        Args:
            optimization_results: List of optimization results
            parameter_grid: Parameter grid used
            optimization_metric: Metric optimized
            
        Returns:
            Optimization report
        """
        
        report = {
            'optimization_date': datetime.now().isoformat(),
            'total_combinations': len(optimization_results),
            'parameter_grid': parameter_grid,
            'optimization_metric': optimization_metric,
            'results': []
        }
        
        # Sort by optimization metric
        sorted_results = sorted(
            optimization_results,
            key=lambda x: x['metric_value'],
            reverse=True
        )
        
        # Top 10 results
        report['top_10_results'] = sorted_results[:10]
        
        # Parameter analysis
        report['parameter_analysis'] = self._analyze_parameters(
            sorted_results, parameter_grid
        )
        
        # Heatmaps for 2D parameter spaces
        report['heatmaps'] = self._generate_parameter_heatmaps(
            sorted_results, parameter_grid, optimization_metric
        )
        
        # Statistical analysis
        report['statistics'] = {
            'best_value': sorted_results[0]['metric_value'],
            'worst_value': sorted_results[-1]['metric_value'],
            'mean_value': np.mean([r['metric_value'] for r in sorted_results]),
            'std_value': np.std([r['metric_value'] for r in sorted_results]),
            'parameter_sensitivity': self._calculate_parameter_sensitivity(
                sorted_results, parameter_grid
            )
        }
        
        return report
        
    def generate_paper_trading_report(self,
                                    performance_data: Dict[str, Any],
                                    trades: List[Dict[str, Any]],
                                    positions: List[Dict[str, Any]],
                                    start_date: datetime,
                                    end_date: datetime) -> Dict[str, Any]:
        """
        Generate paper trading performance report
        
        Args:
            performance_data: Overall performance metrics
            trades: List of executed trades
            positions: Current positions
            start_date: Trading period start
            end_date: Trading period end
            
        Returns:
            Paper trading report
        """
        
        # Calculate metrics from trades
        equity_data = self._build_equity_curve_from_trades(trades, performance_data['initial_capital'])
        metrics = self.metrics_calculator.calculate_metrics(
            trades, equity_data, performance_data['initial_capital']
        )
        
        report = {
            'report_date': datetime.now().isoformat(),
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'trading_days': (end_date - start_date).days
            },
            'account_summary': {
                'initial_capital': performance_data['initial_capital'],
                'current_capital': performance_data['current_capital'],
                'portfolio_value': performance_data['portfolio_value'],
                'total_pnl': performance_data['total_pnl'],
                'total_return_pct': performance_data['total_return_pct']
            },
            'performance_metrics': metrics,
            'position_summary': self._summarize_positions(positions),
            'trade_analysis': self._analyze_trades_detailed(trades),
            'daily_statistics': self._calculate_daily_statistics(trades, equity_data),
            'risk_analysis': self._analyze_risk(trades, equity_data, positions)
        }
        
        return report
        
    def _build_report_data(self, 
                         result: BacktestResult,
                         include_trades: bool,
                         include_equity_curve: bool) -> Dict[str, Any]:
        """Build comprehensive report data structure"""
        
        report_data = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'generator_version': '1.0.0',
                'report_type': 'backtest'
            },
            'backtest_configuration': {
                'strategy': result.strategy_name,
                'symbols': result.symbols,
                'start_date': result.start_date.isoformat(),
                'end_date': result.end_date.isoformat(),
                'initial_capital': result.initial_capital,
                'parameters': result.metadata.get('strategy_parameters', {}),
                'commission': result.metadata.get('commission', 0),
                'slippage': result.metadata.get('slippage', 0)
            },
            'execution_statistics': {
                'execution_time': result.execution_time,
                'bars_processed': result.total_bars_processed,
                'data_frequency': result.metadata.get('data_frequency', 'unknown')
            },
            'performance_summary': {
                'initial_capital': result.initial_capital,
                'final_capital': result.final_capital,
                'total_return': result.metrics.get('total_return', 0),
                'annualized_return': result.metrics.get('annualized_return', 0),
                'sharpe_ratio': result.metrics.get('sharpe_ratio', 0),
                'max_drawdown': result.metrics.get('max_drawdown', 0),
                'win_rate': result.metrics.get('win_rate', 0),
                'profit_factor': result.metrics.get('profit_factor', 0),
                'total_trades': len(result.trades)
            },
            'detailed_metrics': result.metrics,
            'risk_metrics': self._extract_risk_metrics(result.metrics),
            'return_metrics': self._extract_return_metrics(result.metrics),
            'trade_metrics': self._extract_trade_metrics(result.metrics),
            'drawdown_analysis': self._analyze_drawdowns(result.equity_curve),
            'monthly_analysis': self._analyze_monthly_performance(result.equity_curve),
            'symbol_performance': self._analyze_symbol_performance(result.trades)
        }
        
        # Add trade details if requested
        if include_trades:
            report_data['trades'] = self._format_trades(result.trades)
            report_data['trade_distribution'] = self._analyze_trade_distribution(result.trades)
            
        # Add equity curve if requested
        if include_equity_curve:
            report_data['equity_curve'] = self._format_equity_curve(result.equity_curve)
            
        # Add position analysis if available
        if result.positions:
            report_data['final_positions'] = result.positions
            
        return report_data
        
    def _extract_risk_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Extract risk-related metrics"""
        
        risk_metrics = {}
        risk_keys = [
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'omega_ratio',
            'max_drawdown', 'avg_drawdown', 'max_drawdown_duration',
            'var_95', 'var_99', 'cvar_95', 'ulcer_index',
            'annual_volatility', 'daily_volatility', 'downside_deviation',
            'beta', 'alpha', 'tail_ratio'
        ]
        
        for key in risk_keys:
            if key in metrics:
                risk_metrics[key] = metrics[key]
                
        return risk_metrics
        
    def _extract_return_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Extract return-related metrics"""
        
        return_metrics = {}
        return_keys = [
            'total_return', 'annualized_return', 'cagr',
            'avg_daily_return', 'avg_monthly_return',
            'best_day_return', 'worst_day_return',
            'best_month', 'worst_month',
            'positive_months', 'negative_months',
            'skewness', 'kurtosis'
        ]
        
        for key in return_keys:
            if key in metrics:
                return_metrics[key] = metrics[key]
                
        return return_metrics
        
    def _extract_trade_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Extract trade-related metrics"""
        
        trade_metrics = {}
        trade_keys = [
            'total_trades', 'winning_trades', 'losing_trades',
            'win_rate', 'avg_win', 'avg_loss', 'largest_win', 'largest_loss',
            'profit_factor', 'expectancy', 'payoff_ratio',
            'max_consecutive_wins', 'max_consecutive_losses',
            'avg_trade_duration_hours', 'kelly_criterion',
            'sqn', 'total_commission', 'total_slippage'
        ]
        
        for key in trade_keys:
            if key in metrics:
                trade_metrics[key] = metrics[key]
                
        return trade_metrics
        
    def _analyze_drawdowns(self, equity_curve: pd.DataFrame) -> Dict[str, Any]:
        """Analyze drawdown periods"""
        
        if equity_curve.empty or 'portfolio_value' not in equity_curve.columns:
            return {}
            
        values = equity_curve['portfolio_value']
        cummax = values.cummax()
        drawdown = (values - cummax) / cummax
        
        # Find all drawdown periods
        drawdown_periods = []
        in_drawdown = drawdown < 0
        start_idx = None
        
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start_idx is None:
                start_idx = i
            elif not is_dd and start_idx is not None:
                # Record drawdown period
                period_dd = drawdown.iloc[start_idx:i]
                drawdown_periods.append({
                    'start': equity_curve.index[start_idx].isoformat(),
                    'end': equity_curve.index[i-1].isoformat(),
                    'duration_days': (equity_curve.index[i-1] - equity_curve.index[start_idx]).days,
                    'max_drawdown': float(period_dd.min() * 100),
                    'recovery_days': (equity_curve.index[i-1] - equity_curve.index[start_idx]).days
                })
                start_idx = None
                
        # Sort by drawdown magnitude
        drawdown_periods.sort(key=lambda x: x['max_drawdown'])
        
        return {
            'total_drawdown_periods': len(drawdown_periods),
            'largest_drawdowns': drawdown_periods[:5],  # Top 5 drawdowns
            'current_drawdown': float(drawdown.iloc[-1] * 100) if len(drawdown) > 0 else 0,
            'avg_drawdown_duration': np.mean([p['duration_days'] for p in drawdown_periods]) if drawdown_periods else 0,
            'drawdown_frequency': len(drawdown_periods) / max(1, len(equity_curve) / 252)  # Per year
        }
        
    def _analyze_monthly_performance(self, equity_curve: pd.DataFrame) -> Dict[str, Any]:
        """Analyze monthly performance"""
        
        if equity_curve.empty or 'portfolio_value' not in equity_curve.columns:
            return {}
            
        # Resample to monthly
        monthly_values = equity_curve['portfolio_value'].resample('M').last()
        monthly_returns = monthly_values.pct_change().dropna()
        
        if monthly_returns.empty:
            return {}
            
        # Group by year and month
        monthly_data = []
        for date, ret in monthly_returns.items():
            monthly_data.append({
                'year': date.year,
                'month': date.month,
                'return': float(ret * 100),
                'cumulative_return': float(((monthly_values[date] / monthly_values.iloc[0]) - 1) * 100)
            })
            
        # Calculate yearly returns
        yearly_returns = {}
        for year in set(d['year'] for d in monthly_data):
            year_data = [d for d in monthly_data if d['year'] == year]
            yearly_returns[year] = {
                'annual_return': sum(d['return'] for d in year_data),
                'best_month': max(d['return'] for d in year_data),
                'worst_month': min(d['return'] for d in year_data),
                'positive_months': sum(1 for d in year_data if d['return'] > 0),
                'negative_months': sum(1 for d in year_data if d['return'] < 0)
            }
            
        return {
            'monthly_returns': monthly_data,
            'yearly_summary': yearly_returns,
            'all_time_best_month': max(monthly_data, key=lambda x: x['return']),
            'all_time_worst_month': min(monthly_data, key=lambda x: x['return']),
            'positive_months_pct': (monthly_returns > 0).mean() * 100,
            'avg_positive_month': monthly_returns[monthly_returns > 0].mean() * 100 if any(monthly_returns > 0) else 0,
            'avg_negative_month': monthly_returns[monthly_returns < 0].mean() * 100 if any(monthly_returns < 0) else 0
        }
        
    def _analyze_symbol_performance(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance by symbol"""
        
        if not trades:
            return {}
            
        symbol_stats = defaultdict(lambda: {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'total_commission': 0,
            'largest_win': 0,
            'largest_loss': 0
        })
        
        for trade in trades:
            symbol = trade['symbol']
            stats = symbol_stats[symbol]
            
            stats['total_trades'] += 1
            
            if 'pnl' in trade:
                pnl = trade['pnl']
                stats['total_pnl'] += pnl
                
                if pnl > 0:
                    stats['winning_trades'] += 1
                    stats['largest_win'] = max(stats['largest_win'], pnl)
                elif pnl < 0:
                    stats['losing_trades'] += 1
                    stats['largest_loss'] = min(stats['largest_loss'], pnl)
                    
            if 'commission' in trade:
                stats['total_commission'] += trade['commission']
                
        # Calculate additional metrics
        for symbol, stats in symbol_stats.items():
            if stats['total_trades'] > 0:
                stats['win_rate'] = stats['winning_trades'] / stats['total_trades'] * 100
                stats['avg_pnl'] = stats['total_pnl'] / stats['total_trades']
                stats['net_pnl'] = stats['total_pnl'] - stats['total_commission']
                
        # Sort by total P&L
        sorted_symbols = sorted(
            symbol_stats.items(),
            key=lambda x: x[1]['total_pnl'],
            reverse=True
        )
        
        return {
            'symbol_performance': dict(sorted_symbols),
            'best_symbol': sorted_symbols[0][0] if sorted_symbols else None,
            'worst_symbol': sorted_symbols[-1][0] if sorted_symbols else None,
            'total_symbols_traded': len(symbol_stats)
        }
        
    def _format_trades(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format trades for report"""
        
        formatted_trades = []
        
        for i, trade in enumerate(trades):
            formatted_trade = {
                'trade_number': i + 1,
                'symbol': trade['symbol'],
                'side': trade['side'],
                'quantity': trade['quantity'],
                'entry_price': trade.get('price', trade.get('entry_price', 0)),
                'exit_price': trade.get('exit_price'),
                'entry_time': self._format_datetime(trade.get('timestamp', trade.get('entry_time'))),
                'exit_time': self._format_datetime(trade.get('exit_time')),
                'pnl': trade.get('pnl', 0),
                'pnl_pct': trade.get('pnl_pct', 0),
                'commission': trade.get('commission', 0),
                'duration_hours': self._calculate_trade_duration(trade)
            }
            
            # Add metadata if available
            if 'metadata' in trade:
                formatted_trade['metadata'] = trade['metadata']
                
            formatted_trades.append(formatted_trade)
            
        return formatted_trades
        
    def _analyze_trade_distribution(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trade distribution"""
        
        if not trades:
            return {}
            
        # P&L distribution
        pnls = [t.get('pnl', 0) for t in trades if 'pnl' in t]
        
        if not pnls:
            return {}
            
        # Calculate bins for histogram
        pnl_bins = np.histogram(pnls, bins=20)
        
        # Time distribution
        trade_hours = []
        trade_days = []
        
        for trade in trades:
            if 'timestamp' in trade:
                dt = pd.to_datetime(trade['timestamp'])
                trade_hours.append(dt.hour)
                trade_days.append(dt.day_name())
                
        return {
            'pnl_distribution': {
                'bins': pnl_bins[1].tolist(),
                'counts': pnl_bins[0].tolist(),
                'mean': float(np.mean(pnls)),
                'median': float(np.median(pnls)),
                'std': float(np.std(pnls)),
                'percentiles': {
                    '5th': float(np.percentile(pnls, 5)),
                    '25th': float(np.percentile(pnls, 25)),
                    '50th': float(np.percentile(pnls, 50)),
                    '75th': float(np.percentile(pnls, 75)),
                    '95th': float(np.percentile(pnls, 95))
                }
            },
            'time_distribution': {
                'by_hour': dict(pd.Series(trade_hours).value_counts().sort_index()),
                'by_day': dict(pd.Series(trade_days).value_counts())
            },
            'trade_size_distribution': self._analyze_trade_sizes(trades)
        }
        
    def _format_equity_curve(self, equity_curve: pd.DataFrame) -> List[Dict[str, Any]]:
        """Format equity curve for report"""
        
        if equity_curve.empty:
            return []
            
        formatted_curve = []
        
        for timestamp, row in equity_curve.iterrows():
            point = {
                'timestamp': self._format_datetime(timestamp),
                'portfolio_value': float(row.get('portfolio_value', 0)),
                'returns': float(row.get('returns', 0) * 100),
                'cumulative_returns': float(row.get('cumulative_returns', 0) * 100),
                'drawdown': float(row.get('drawdown', 0) * 100)
            }
            
            # Add additional fields if available
            for field in ['cash_balance', 'positions_value', 'daily_pnl']:
                if field in row:
                    point[field] = float(row[field])
                    
            formatted_curve.append(point)
            
        return formatted_curve
        
    def _generate_json_report(self, report_data: Dict[str, Any]) -> str:
        """Generate JSON format report"""
        
        return json.dumps(report_data, indent=2, default=str)
        
    def _generate_csv_report(self, report_data: Dict[str, Any]) -> str:
        """Generate CSV format report"""
        
        output = io.StringIO()
        
        # Write summary metrics
        output.write("Metric,Value\n")
        for key, value in report_data['performance_summary'].items():
            output.write(f"{key},{value}\n")
            
        output.write("\n")
        
        # Write trades if available
        if 'trades' in report_data and report_data['trades']:
            trades_df = pd.DataFrame(report_data['trades'])
            output.write("\nTrades\n")
            trades_df.to_csv(output, index=False)
            
        return output.getvalue()
        
    def _generate_excel_report(self, report_data: Dict[str, Any]) -> bytes:
        """Generate Excel format report"""
        
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Summary sheet
            summary_df = pd.DataFrame([report_data['performance_summary']])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed metrics sheet
            metrics_df = pd.DataFrame([report_data['detailed_metrics']])
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            
            # Trades sheet
            if 'trades' in report_data and report_data['trades']:
                trades_df = pd.DataFrame(report_data['trades'])
                trades_df.to_excel(writer, sheet_name='Trades', index=False)
                
            # Equity curve sheet
            if 'equity_curve' in report_data and report_data['equity_curve']:
                equity_df = pd.DataFrame(report_data['equity_curve'])
                equity_df.to_excel(writer, sheet_name='Equity Curve', index=False)
                
            # Monthly analysis sheet
            if 'monthly_analysis' in report_data and 'monthly_returns' in report_data['monthly_analysis']:
                monthly_df = pd.DataFrame(report_data['monthly_analysis']['monthly_returns'])
                monthly_df.to_excel(writer, sheet_name='Monthly Returns', index=False)
                
            # Symbol performance sheet
            if 'symbol_performance' in report_data and report_data['symbol_performance']:
                symbol_data = []
                for symbol, stats in report_data['symbol_performance']['symbol_performance'].items():
                    stats['symbol'] = symbol
                    symbol_data.append(stats)
                symbol_df = pd.DataFrame(symbol_data)
                symbol_df.to_excel(writer, sheet_name='Symbol Performance', index=False)
                
            # Add charts
            self._add_excel_charts(writer, report_data)
            
        output.seek(0)
        return output.getvalue()
        
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML format report"""
        
        # Load template
        template_path = self.template_dir / "backtest_report.html"
        
        if template_path.exists():
            with open(template_path, 'r') as f:
                template = Template(f.read())
        else:
            # Use default template
            template = Template(self._get_default_html_template())
            
        # Prepare data for template
        context = {
            'report_data': report_data,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'charts': self._generate_chart_data(report_data)
        }
        
        return template.render(**context)
        
    def _generate_pdf_report(self, report_data: Dict[str, Any]) -> bytes:
        """Generate PDF format report"""
        
        # First generate HTML
        html_content = self._generate_html_report(report_data)
        
        # Convert HTML to PDF using weasyprint or similar
        # Note: This requires additional dependencies
        try:
            from weasyprint import HTML
            pdf_content = HTML(string=html_content).write_pdf()
            return pdf_content
        except ImportError:
            logger.warning("PDF generation requires weasyprint. Install with: pip install weasyprint")
            # Return HTML as fallback
            return html_content.encode('utf-8')
        
    def _rank_strategies(self, strategies: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Rank strategies by different metrics"""
        
        ranking_metrics = [
            'total_return',
            'sharpe_ratio',
            'calmar_ratio',
            'profit_factor',
            'win_rate',
            'sqn'
        ]
        
        rankings = {}
        
        for metric in ranking_metrics:
            # Sort strategies by metric
            sorted_strategies = sorted(
                strategies,
                key=lambda x: x['summary'].get(metric, -float('inf')),
                reverse=True
            )
            
            # Create ranking with rank number
            rankings[metric] = []
            for i, strategy in enumerate(sorted_strategies):
                rankings[metric].append({
                    'rank': i + 1,
                    'name': strategy['name'],
                    'value': strategy['summary'].get(metric, 0)
                })
                
        return rankings
        
    def _compare_statistics(self, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistical comparison of strategies"""
        
        if not strategies:
            return {}
            
        # Extract metrics for comparison
        metrics_data = defaultdict(list)
        
        for strategy in strategies:
            for metric, value in strategy['summary'].items():
                if isinstance(value, (int, float)):
                    metrics_data[metric].append(value)
                    
        # Calculate statistics for each metric
        statistics = {}
        
        for metric, values in metrics_data.items():
            if values:
                statistics[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(min(values)),
                    'max': float(max(values)),
                    'range': float(max(values) - min(values)),
                    'cv': float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0
                }
                
        return statistics
        
    def _analyze_parameters(self, 
                          results: List[Dict[str, Any]], 
                          parameter_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Analyze parameter impact on performance"""
        
        parameter_analysis = {}
        
        for param_name, param_values in parameter_grid.items():
            # Group results by parameter value
            param_groups = defaultdict(list)
            
            for result in results:
                param_value = result['parameters'].get(param_name)
                param_groups[param_value].append(result['metric_value'])
                
            # Calculate statistics for each parameter value
            param_stats = {}
            
            for value, metrics in param_groups.items():
                param_stats[str(value)] = {
                    'mean': float(np.mean(metrics)),
                    'std': float(np.std(metrics)),
                    'count': len(metrics),
                    'min': float(min(metrics)),
                    'max': float(max(metrics))
                }
                
            parameter_analysis[param_name] = {
                'values': param_stats,
                'best_value': max(param_stats.keys(), key=lambda x: param_stats[x]['mean']),
                'worst_value': min(param_stats.keys(), key=lambda x: param_stats[x]['mean']),
                'sensitivity': float(np.std([s['mean'] for s in param_stats.values()]))
            }
            
        return parameter_analysis
        
    def _generate_parameter_heatmaps(self,
                                   results: List[Dict[str, Any]],
                                   parameter_grid: Dict[str, List[Any]],
                                   metric: str) -> Dict[str, Any]:
        """Generate heatmap data for 2D parameter spaces"""
        
        heatmaps = {}
        param_names = list(parameter_grid.keys())
        
        # Generate heatmaps for each pair of parameters
        for i in range(len(param_names)):
            for j in range(i + 1, len(param_names)):
                param1 = param_names[i]
                param2 = param_names[j]
                
                # Create 2D matrix
                values1 = parameter_grid[param1]
                values2 = parameter_grid[param2]
                
                heatmap = np.zeros((len(values1), len(values2)))
                counts = np.zeros((len(values1), len(values2)))
                
                # Fill matrix with metric values
                for result in results:
                    idx1 = values1.index(result['parameters'][param1])
                    idx2 = values2.index(result['parameters'][param2])
                    heatmap[idx1, idx2] += result['metric_value']
                    counts[idx1, idx2] += 1
                    
                # Average values
                with np.errstate(divide='ignore', invalid='ignore'):
                    heatmap = np.divide(heatmap, counts)
                    heatmap[counts == 0] = np.nan
                    
                heatmaps[f"{param1}_vs_{param2}"] = {
                    'x_param': param1,
                    'y_param': param2,
                    'x_values': values1,
                    'y_values': values2,
                    'values': heatmap.tolist(),
                    'metric': metric
                }
                
        return heatmaps
        
    def _calculate_parameter_sensitivity(self,
                                       results: List[Dict[str, Any]],
                                       parameter_grid: Dict[str, List[Any]]) -> Dict[str, float]:
        """Calculate sensitivity of performance to each parameter"""
        
        sensitivities = {}
        
        for param_name in parameter_grid.keys():
            # Group results by parameter value
            param_groups = defaultdict(list)
            
            for result in results:
                param_value = result['parameters'].get(param_name)
                param_groups[param_value].append(result['metric_value'])
                
            # Calculate variance across parameter values
            group_means = [np.mean(metrics) for metrics in param_groups.values()]
            
            if len(group_means) > 1:
                sensitivities[param_name] = float(np.std(group_means))
            else:
                sensitivities[param_name] = 0.0
                
        # Normalize sensitivities
        total_sensitivity = sum(sensitivities.values())
        if total_sensitivity > 0:
            sensitivities = {k: v / total_sensitivity for k, v in sensitivities.items()}
            
        return sensitivities
        
    def _build_equity_curve_from_trades(self, 
                                      trades: List[Dict[str, Any]], 
                                      initial_capital: float) -> pd.DataFrame:
        """Build equity curve from trades"""
        
        if not trades:
            return pd.DataFrame()
            
        # Sort trades by timestamp
        sorted_trades = sorted(trades, key=lambda x: x.get('timestamp', datetime.now()))
        
        # Build equity points
        equity_points = []
        current_capital = initial_capital
        
        for trade in sorted_trades:
            if 'pnl' in trade:
                current_capital += trade['pnl']
                
            equity_points.append({
                'timestamp': trade.get('timestamp', datetime.now()),
                'portfolio_value': current_capital
            })
            
        # Create DataFrame
        df = pd.DataFrame(equity_points)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
            
        return df
        
    def _summarize_positions(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize current positions"""
        
        if not positions:
            return {
                'total_positions': 0,
                'total_value': 0,
                'total_unrealized_pnl': 0
            }
            
        total_value = sum(p.get('market_value', 0) for p in positions)
        total_unrealized_pnl = sum(p.get('unrealized_pnl', 0) for p in positions)
        
        # Group by symbol
        by_symbol = defaultdict(lambda: {'quantity': 0, 'value': 0, 'pnl': 0})
        
        for position in positions:
            symbol = position['symbol']
            by_symbol[symbol]['quantity'] += position.get('quantity', 0)
            by_symbol[symbol]['value'] += position.get('market_value', 0)
            by_symbol[symbol]['pnl'] += position.get('unrealized_pnl', 0)
            
        return {
            'total_positions': len(positions),
            'total_value': total_value,
            'total_unrealized_pnl': total_unrealized_pnl,
            'average_position_size': total_value / len(positions) if positions else 0,
            'largest_position': max(positions, key=lambda x: x.get('market_value', 0)),
            'positions_by_symbol': dict(by_symbol)
        }
        
    def _analyze_trades_detailed(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detailed trade analysis"""
        
        if not trades:
            return {}
            
        # Time-based analysis
        trades_by_hour = defaultdict(int)
        trades_by_day = defaultdict(int)
        trades_by_month = defaultdict(int)
        
        for trade in trades:
            if 'timestamp' in trade:
                dt = pd.to_datetime(trade['timestamp'])
                trades_by_hour[dt.hour] += 1
                trades_by_day[dt.day_name()] += 1
                trades_by_month[f"{dt.year}-{dt.month:02d}"] += 1
                
        # Duration analysis
        durations = []
        for trade in trades:
            duration = self._calculate_trade_duration(trade)
            if duration is not None:
                durations.append(duration)
                
        return {
            'temporal_distribution': {
                'by_hour': dict(trades_by_hour),
                'by_day': dict(trades_by_day),
                'by_month': dict(trades_by_month)
            },
            'duration_analysis': {
                'avg_duration_hours': float(np.mean(durations)) if durations else 0,
                'median_duration_hours': float(np.median(durations)) if durations else 0,
                'min_duration_hours': float(min(durations)) if durations else 0,
                'max_duration_hours': float(max(durations)) if durations else 0
            },
            'trade_sequence_analysis': self._analyze_trade_sequences(trades)
        }
        
    def _calculate_daily_statistics(self, 
                                  trades: List[Dict[str, Any]], 
                                  equity_curve: pd.DataFrame) -> Dict[str, Any]:
        """Calculate daily trading statistics"""
        
        if not trades:
            return {}
            
        # Group trades by date
        trades_by_date = defaultdict(list)
        
        for trade in trades:
            if 'timestamp' in trade:
                date = pd.to_datetime(trade['timestamp']).date()
                trades_by_date[date].append(trade)
                
        # Calculate daily stats
        daily_stats = []
        
        for date, day_trades in trades_by_date.items():
            daily_pnl = sum(t.get('pnl', 0) for t in day_trades)
            
            daily_stats.append({
                'date': date.isoformat(),
                'trades': len(day_trades),
                'pnl': daily_pnl,
                'win_rate': sum(1 for t in day_trades if t.get('pnl', 0) > 0) / len(day_trades) * 100 if day_trades else 0
            })
            
        return {
            'daily_statistics': daily_stats,
            'avg_trades_per_day': float(np.mean([s['trades'] for s in daily_stats])) if daily_stats else 0,
            'best_day': max(daily_stats, key=lambda x: x['pnl']) if daily_stats else None,
            'worst_day': min(daily_stats, key=lambda x: x['pnl']) if daily_stats else None
        }
        
    def _analyze_risk(self, 
                     trades: List[Dict[str, Any]], 
                     equity_curve: pd.DataFrame,
                     positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Comprehensive risk analysis"""
        
        risk_analysis = {
            'position_concentration': self._analyze_position_concentration(positions),
            'trade_risk': self._analyze_trade_risk(trades),
            'drawdown_risk': self._analyze_drawdown_risk(equity_curve),
            'correlation_risk': self._analyze_correlation_risk(trades)
        }
        
        return risk_analysis
        
    def _analyze_position_concentration(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze position concentration risk"""
        
        if not positions:
            return {}
            
        total_value = sum(p.get('market_value', 0) for p in positions)
        
        if total_value == 0:
            return {}
            
        # Calculate concentration metrics
        position_weights = []
        
        for position in positions:
            weight = position.get('market_value', 0) / total_value
            position_weights.append({
                'symbol': position['symbol'],
                'weight': weight * 100,
                'value': position.get('market_value', 0)
            })
            
        # Sort by weight
        position_weights.sort(key=lambda x: x['weight'], reverse=True)
        
        # Calculate HHI (Herfindahl-Hirschman Index)
        hhi = sum((w['weight'] / 100) ** 2 for w in position_weights) * 10000
        
        return {
            'position_weights': position_weights[:10],  # Top 10
            'herfindahl_index': hhi,
            'largest_position_pct': position_weights[0]['weight'] if position_weights else 0,
            'top_5_concentration_pct': sum(w['weight'] for w in position_weights[:5])
        }
        
    def _analyze_trade_risk(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trade-based risk metrics"""
        
        if not trades:
            return {}
            
        pnls = [t.get('pnl', 0) for t in trades if 'pnl' in t]
        
        if not pnls:
            return {}
            
        # Risk metrics
        downside_pnls = [p for p in pnls if p < 0]
        
        return {
            'avg_loss_per_trade': float(np.mean(downside_pnls)) if downside_pnls else 0,
            'max_loss_per_trade': float(min(pnls)),
            'loss_standard_deviation': float(np.std(downside_pnls)) if downside_pnls else 0,
            'downside_deviation': float(np.std([min(0, p) for p in pnls])),
            'semi_variance': float(np.var([p for p in pnls if p < 0])) if downside_pnls else 0
        }
        
    def _analyze_drawdown_risk(self, equity_curve: pd.DataFrame) -> Dict[str, Any]:
        """Analyze drawdown-based risk"""
        
        if equity_curve.empty or 'portfolio_value' not in equity_curve.columns:
            return {}
            
        values = equity_curve['portfolio_value']
        cummax = values.cummax()
        drawdown = (values - cummax) / cummax
        
        # Drawdown statistics
        drawdown_values = drawdown[drawdown < 0]
        
        if len(drawdown_values) == 0:
            return {}
            
        return {
            'current_drawdown_pct': float(drawdown.iloc[-1] * 100),
            'average_drawdown_pct': float(drawdown_values.mean() * 100),
            'drawdown_volatility': float(drawdown_values.std() * 100),
            'time_in_drawdown_pct': float(len(drawdown_values) / len(drawdown) * 100),
            'median_drawdown_pct': float(drawdown_values.median() * 100)
        }
        
    def _analyze_correlation_risk(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze correlation risk between trades"""
        
        # Simplified correlation analysis
        # In production, would calculate actual correlations between symbols
        
        return {
            'note': 'Correlation analysis requires market data',
            'recommendation': 'Monitor correlations between traded symbols'
        }
        
    def _analyze_trade_sequences(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trade sequences and patterns"""
        
        if not trades:
            return {}
            
        # Analyze win/loss sequences
        sequences = []
        current_sequence = []
        current_type = None
        
        for trade in trades:
            if 'pnl' in trade:
                trade_type = 'win' if trade['pnl'] > 0 else 'loss'
                
                if trade_type == current_type:
                    current_sequence.append(trade)
                else:
                    if current_sequence:
                        sequences.append({
                            'type': current_type,
                            'length': len(current_sequence),
                            'total_pnl': sum(t['pnl'] for t in current_sequence)
                        })
                    current_sequence = [trade]
                    current_type = trade_type
                    
        # Add final sequence
        if current_sequence:
            sequences.append({
                'type': current_type,
                'length': len(current_sequence),
                'total_pnl': sum(t['pnl'] for t in current_sequence)
            })
            
        # Analyze sequences
        win_sequences = [s for s in sequences if s['type'] == 'win']
        loss_sequences = [s for s in sequences if s['type'] == 'loss']
        
        return {
            'total_sequences': len(sequences),
            'win_sequences': len(win_sequences),
            'loss_sequences': len(loss_sequences),
            'longest_win_streak': max([s['length'] for s in win_sequences]) if win_sequences else 0,
            'longest_loss_streak': max([s['length'] for s in loss_sequences]) if loss_sequences else 0,
            'avg_win_streak': float(np.mean([s['length'] for s in win_sequences])) if win_sequences else 0,
            'avg_loss_streak': float(np.mean([s['length'] for s in loss_sequences])) if loss_sequences else 0
        }
        
    def _analyze_trade_sizes(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trade size distribution"""
        
        if not trades:
            return {}
            
        trade_values = []
        
        for trade in trades:
            if 'quantity' in trade and 'price' in trade:
                trade_values.append(trade['quantity'] * trade.get('price', 0))
            elif 'entry_price' in trade and 'quantity' in trade:
                trade_values.append(trade['quantity'] * trade['entry_price'])
                
        if not trade_values:
            return {}
            
        return {
            'avg_trade_size': float(np.mean(trade_values)),
            'median_trade_size': float(np.median(trade_values)),
            'min_trade_size': float(min(trade_values)),
            'max_trade_size': float(max(trade_values)),
            'trade_size_std': float(np.std(trade_values))
        }
        
    def _format_datetime(self, dt: Any) -> str:
        """Format datetime for report"""
        
        if dt is None:
            return ""
            
        if isinstance(dt, str):
            return dt
            
        if isinstance(dt, (datetime, pd.Timestamp)):
            return dt.strftime('%Y-%m-%d %H:%M:%S')
            
        return str(dt)
        
    def _calculate_trade_duration(self, trade: Dict[str, Any]) -> Optional[float]:
        """Calculate trade duration in hours"""
        
        entry_time = None
        exit_time = None
        
        if 'entry_time' in trade:
            entry_time = pd.to_datetime(trade['entry_time'])
        elif 'timestamp' in trade:
            entry_time = pd.to_datetime(trade['timestamp'])
            
        if 'exit_time' in trade and trade['exit_time']:
            exit_time = pd.to_datetime(trade['exit_time'])
            
        if entry_time and exit_time:
            return (exit_time - entry_time).total_seconds() / 3600
            
        return None
        
    def _generate_chart_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data for charts"""
        
        charts = {}
        
        # Equity curve chart
        if 'equity_curve' in report_data:
            charts['equity_curve'] = {
                'type': 'line',
                'data': report_data['equity_curve'],
                'x_field': 'timestamp',
                'y_field': 'portfolio_value',
                'title': 'Portfolio Value Over Time'
            }
            
        # Monthly returns chart
        if 'monthly_analysis' in report_data and 'monthly_returns' in report_data['monthly_analysis']:
            charts['monthly_returns'] = {
                'type': 'bar',
                'data': report_data['monthly_analysis']['monthly_returns'],
                'x_field': 'month',
                'y_field': 'return',
                'title': 'Monthly Returns'
            }
            
        # Drawdown chart
        if 'equity_curve' in report_data:
            charts['drawdown'] = {
                'type': 'area',
                'data': report_data['equity_curve'],
                'x_field': 'timestamp',
                'y_field': 'drawdown',
                'title': 'Drawdown Over Time'
            }
            
        # Win/Loss distribution
        if 'trade_distribution' in report_data and 'pnl_distribution' in report_data['trade_distribution']:
            charts['pnl_distribution'] = {
                'type': 'histogram',
                'data': report_data['trade_distribution']['pnl_distribution'],
                'title': 'P&L Distribution'
            }
            
        return charts
        
    def _add_excel_charts(self, writer: pd.ExcelWriter, report_data: Dict[str, Any]):
        """Add charts to Excel report"""
        
        workbook = writer.book
        
        # Equity curve chart
        if 'equity_curve' in report_data and report_data['equity_curve']:
            worksheet = writer.sheets['Equity Curve']
            
            chart = workbook.add_chart({'type': 'line'})
            chart.add_series({
                'categories': ['Equity Curve', 1, 0, len(report_data['equity_curve']), 0],
                'values': ['Equity Curve', 1, 1, len(report_data['equity_curve']), 1],
                'name': 'Portfolio Value'
            })
            
            chart.set_title({'name': 'Portfolio Value Over Time'})
            chart.set_x_axis({'name': 'Date'})
            chart.set_y_axis({'name': 'Value'})
            
            worksheet.insert_chart('E2', chart)
            
    def _get_default_html_template(self) -> str:
        """Get default HTML template"""
        
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2, h3 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .metric { margin: 10px 0; }
        .metric-label { font-weight: bold; }
        .metric-value { color: #007bff; }
        .positive { color: green; }
        .negative { color: red; }
    </style>
</head>
<body>
    <h1>Backtest Report</h1>
    <p>Generated at: {{ generated_at }}</p>
    
    <h2>Performance Summary</h2>
    <div class="metrics">
        {% for key, value in report_data.performance_summary.items() %}
        <div class="metric">
            <span class="metric-label">{{ key }}:</span>
            <span class="metric-value {% if value > 0 %}positive{% elif value < 0 %}negative{% endif %}">
                {{ "%.2f"|format(value) }}{% if key.endswith('_pct') or key.endswith('rate') %}%{% endif %}
            </span>
        </div>
        {% endfor %}
    </div>
    
    <h2>Risk Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        {% for key, value in report_data.risk_metrics.items() %}
        <tr>
            <td>{{ key }}</td>
            <td>{{ "%.4f"|format(value) }}</td>
        </tr>
        {% endfor %}
    </table>
    
    {% if report_data.trades %}
    <h2>Trade Analysis</h2>
    <table>
        <tr>
            <th>Trade #</th>
            <th>Symbol</th>
            <th>Side</th>
            <th>Entry</th>
            <th>Exit</th>
            <th>P&L</th>
        </tr>
        {% for trade in report_data.trades[:20] %}
        <tr>
            <td>{{ trade.trade_number }}</td>
            <td>{{ trade.symbol }}</td>
            <td>{{ trade.side }}</td>
            <td>{{ "%.2f"|format(trade.entry_price) }}</td>
            <td>{{ "%.2f"|format(trade.exit_price or 0) }}</td>
            <td class="{% if trade.pnl > 0 %}positive{% else %}negative{% endif %}">
                {{ "%.2f"|format(trade.pnl) }}
            </td>
        </tr>
        {% endfor %}
    </table>
    {% endif %}
</body>
</html>
"""
        
    def _generate_comparison_html(self, comparison_data: Dict[str, Any]) -> str:
        """Generate HTML for strategy comparison"""
        
        template = Template("""
<!DOCTYPE html>
<html>
<head>
    <title>Strategy Comparison Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .best { background-color: #d4edda; }
        .worst { background-color: #f8d7da; }
    </style>
</head>
<body>
    <h1>Strategy Comparison Report</h1>
    <p>Total Strategies: {{ comparison_data.total_backtests }}</p>
    
    <h2>Performance Comparison</h2>
    <table>
        <tr>
            <th>Strategy</th>
            <th>Total Return %</th>
            <th>Sharpe Ratio</th>
            <th>Max Drawdown %</th>
            <th>Win Rate %</th>
            <th>Total Trades</th>
        </tr>
        {% for strategy in comparison_data.strategies %}
        <tr>
            <td>{{ strategy.name }}</td>
            <td>{{ "%.2f"|format(strategy.summary.total_return) }}</td>
            <td>{{ "%.3f"|format(strategy.summary.sharpe_ratio) }}</td>
            <td>{{ "%.2f"|format(strategy.summary.max_drawdown) }}</td>
            <td>{{ "%.1f"|format(strategy.summary.win_rate) }}</td>
            <td>{{ strategy.summary.total_trades }}</td>
        </tr>
        {% endfor %}
    </table>
    
    <h2>Rankings</h2>
    {% for metric, ranking in comparison_data.rankings.items() %}
    <h3>{{ metric }}</h3>
    <ol>
        {% for item in ranking[:5] %}
        <li>{{ item.name }}: {{ "%.3f"|format(item.value) }}</li>
        {% endfor %}
    </ol>
    {% endfor %}
</body>
</html>
""")
        
        return template.render(comparison_data=comparison_data)
        
    def _save_report(self, report: Union[str, bytes, Dict], output_path: str, format: str):
        """Save report to file"""
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format in ['json']:
            with open(path, 'w') as f:
                if isinstance(report, dict):
                    json.dump(report, f, indent=2, default=str)
                else:
                    f.write(report)
                    
        elif format in ['csv', 'html']:
            with open(path, 'w') as f:
                f.write(report)
                
        elif format in ['excel', 'pdf']:
            with open(path, 'wb') as f:
                f.write(report)
                
        logger.info(f"Report saved to {output_path}")