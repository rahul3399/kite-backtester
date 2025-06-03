# app/backtesting/optimizer.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
from dataclasses import dataclass
import json
import pickle

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Container for optimization results"""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    optimization_metric: str
    convergence_history: List[float]
    parameter_importance: Dict[str, float]
    execution_time: float
    total_iterations: int

class StrategyOptimizer:
    """
    Advanced strategy parameter optimizer with multiple optimization methods
    """
    
    def __init__(self, objective_function: Callable, n_jobs: int = -1):
        """
        Initialize optimizer
        
        Args:
            objective_function: Function that takes parameters and returns score
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.objective_function = objective_function
        self.n_jobs = n_jobs if n_jobs > 0 else None
        self.results_cache = {}
        
    def grid_search(self,
                   param_grid: Dict[str, List[Any]],
                   optimization_metric: str = "sharpe_ratio") -> OptimizationResult:
        """
        Exhaustive grid search optimization
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values
            optimization_metric: Metric to optimize
            
        Returns:
            OptimizationResult object
        """
        
        start_time = datetime.now()
        
        # Generate all parameter combinations
        param_combinations = self._generate_grid_combinations(param_grid)
        total_combinations = len(param_combinations)
        
        logger.info(f"Starting grid search with {total_combinations} combinations")
        
        # Run optimization
        all_results = []
        best_score = -float('inf')
        best_params = None
        convergence_history = []
        
        # Process in batches for parallel execution
        batch_size = max(1, total_combinations // (self.n_jobs or 4))
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit tasks
            future_to_params = {}
            for params in param_combinations:
                future = executor.submit(self._evaluate_parameters, params, optimization_metric)
                future_to_params[future] = params
                
            # Process results as they complete
            for i, future in enumerate(as_completed(future_to_params)):
                params = future_to_params[future]
                
                try:
                    score = future.result()
                    
                    result = {
                        "parameters": params,
                        "score": score,
                        "iteration": i + 1
                    }
                    all_results.append(result)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        
                    convergence_history.append(best_score)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Progress: {i + 1}/{total_combinations} - Best score: {best_score:.4f}")
                        
                except Exception as e:
                    logger.error(f"Error evaluating parameters {params}: {e}")
                    
        # Calculate parameter importance
        parameter_importance = self._calculate_parameter_importance(all_results, param_grid)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_metric=optimization_metric,
            convergence_history=convergence_history,
            parameter_importance=parameter_importance,
            execution_time=execution_time,
            total_iterations=len(all_results)
        )
        
    def random_search(self,
                     param_distributions: Dict[str, Any],
                     n_iter: int = 100,
                     optimization_metric: str = "sharpe_ratio") -> OptimizationResult:
        """
        Random search optimization
        
        Args:
            param_distributions: Parameter distributions (can be lists or scipy distributions)
            n_iter: Number of iterations
            optimization_metric: Metric to optimize
            
        Returns:
            OptimizationResult object
        """
        
        start_time = datetime.now()
        
        # Generate random parameter combinations
        param_combinations = self._generate_random_combinations(param_distributions, n_iter)
        
        logger.info(f"Starting random search with {n_iter} iterations")
        
        # Run optimization
        all_results = []
        best_score = -float('inf')
        best_params = None
        convergence_history = []
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            
            for params in param_combinations:
                future = executor.submit(self._evaluate_parameters, params, optimization_metric)
                futures.append((params, future))
                
            for i, (params, future) in enumerate(futures):
                try:
                    score = future.result()
                    
                    result = {
                        "parameters": params,
                        "score": score,
                        "iteration": i + 1
                    }
                    all_results.append(result)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        
                    convergence_history.append(best_score)
                    
                except Exception as e:
                    logger.error(f"Error evaluating parameters {params}: {e}")
                    
        # Calculate parameter importance
        parameter_importance = self._calculate_parameter_importance_random(all_results)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_metric=optimization_metric,
            convergence_history=convergence_history,
            parameter_importance=parameter_importance,
            execution_time=execution_time,
            total_iterations=len(all_results)
        )
        
    def bayesian_optimization(self,
                            param_bounds: Dict[str, Tuple[float, float]],
                            n_iter: int = 50,
                            optimization_metric: str = "sharpe_ratio",
                            init_points: int = 10) -> OptimizationResult:
        """
        Bayesian optimization using Gaussian Process
        
        Args:
            param_bounds: Parameter bounds as {name: (min, max)}
            n_iter: Number of iterations
            optimization_metric: Metric to optimize
            init_points: Number of initial random points
            
        Returns:
            OptimizationResult object
        """
        
        # This is a simplified implementation
        # In production, use libraries like scikit-optimize or hyperopt
        
        start_time = datetime.now()
        
        # Convert bounds to arrays
        param_names = list(param_bounds.keys())
        bounds = np.array([param_bounds[name] for name in param_names])
        
        # Initial random sampling
        logger.info(f"Starting Bayesian optimization with {init_points} initial points")
        
        all_results = []
        best_score = -float('inf')
        best_params = None
        convergence_history = []
        
        # Random initial points
        for i in range(init_points):
            params = {}
            for j, name in enumerate(param_names):
                low, high = bounds[j]
                if isinstance(low, int) and isinstance(high, int):
                    params[name] = np.random.randint(low, high + 1)
                else:
                    params[name] = np.random.uniform(low, high)
                    
            score = self._evaluate_parameters(params, optimization_metric)
            
            result = {
                "parameters": params,
                "score": score,
                "iteration": i + 1
            }
            all_results.append(result)
            
            if score > best_score:
                best_score = score
                best_params = params
                
            convergence_history.append(best_score)
            
        # Bayesian optimization iterations
        # Simplified - just doing more intelligent random search
        for i in range(init_points, n_iter):
            # In real implementation, use Gaussian Process to predict next point
            # Here, we sample around best point with decreasing variance
            
            variance = 1.0 - (i / n_iter)  # Decrease exploration over time
            
            params = {}
            for j, name in enumerate(param_names):
                low, high = bounds[j]
                center = best_params[name]
                
                # Sample around best with gaussian noise
                if isinstance(low, int) and isinstance(high, int):
                    noise = int(np.random.normal(0, variance * (high - low) / 4))
                    value = int(np.clip(center + noise, low, high))
                    params[name] = value
                else:
                    noise = np.random.normal(0, variance * (high - low) / 4)
                    value = np.clip(center + noise, low, high)
                    params[name] = value
                    
            score = self._evaluate_parameters(params, optimization_metric)
            
            result = {
                "parameters": params,
                "score": score,
                "iteration": i + 1
            }
            all_results.append(result)
            
            if score > best_score:
                best_score = score
                best_params = params
                
            convergence_history.append(best_score)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Iteration {i + 1}/{n_iter} - Best score: {best_score:.4f}")
                
        # Calculate parameter importance
        parameter_importance = self._calculate_parameter_importance_random(all_results)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_metric=optimization_metric,
            convergence_history=convergence_history,
            parameter_importance=parameter_importance,
            execution_time=execution_time,
            total_iterations=len(all_results)
        )
        
    def walk_forward_optimization(self,
                                param_grid: Dict[str, List[Any]],
                                train_periods: int,
                                test_periods: int,
                                step_size: int,
                                optimization_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """
        Walk-forward optimization for robust parameter selection
        
        Args:
            param_grid: Parameter grid
            train_periods: Number of periods for training
            test_periods: Number of periods for testing
            step_size: Step size for moving window
            optimization_metric: Metric to optimize
            
        Returns:
            Walk-forward analysis results
        """
        
        logger.info("Starting walk-forward optimization")
        
        # This would require modification of objective function to accept date ranges
        # Simplified implementation here
        
        windows = []
        window_results = []
        
        # Perform optimization for each window
        # In real implementation, this would slice the data
        
        # For now, just do standard optimization
        result = self.grid_search(param_grid, optimization_metric)
        
        return {
            "final_parameters": result.best_params,
            "average_score": result.best_score,
            "window_results": window_results,
            "parameter_stability": self._calculate_parameter_stability(window_results)
        }
        
    def _evaluate_parameters(self, params: Dict[str, Any], metric: str) -> float:
        """Evaluate parameters and return score"""
        
        # Check cache
        cache_key = json.dumps(params, sort_keys=True)
        if cache_key in self.results_cache:
            return self.results_cache[cache_key]
            
        try:
            # Call objective function
            score = self.objective_function(params, metric)
            
            # Cache result
            self.results_cache[cache_key] = score
            
            return score
            
        except Exception as e:
            logger.error(f"Error evaluating parameters {params}: {e}")
            return -float('inf')
            
    def _generate_grid_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from grid"""
        
        keys = list(param_grid.keys())
        values = [param_grid[key] for key in keys]
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
            
        return combinations
        
    def _generate_random_combinations(self, 
                                    param_distributions: Dict[str, Any], 
                                    n_iter: int) -> List[Dict[str, Any]]:
        """Generate random parameter combinations"""
        
        combinations = []
        
        for _ in range(n_iter):
            params = {}
            
            for name, distribution in param_distributions.items():
                if isinstance(distribution, list):
                    # Sample from list
                    params[name] = np.random.choice(distribution)
                elif hasattr(distribution, 'rvs'):
                    # Scipy distribution
                    params[name] = distribution.rvs()
                elif isinstance(distribution, tuple) and len(distribution) == 2:
                    # Range (min, max)
                    low, high = distribution
                    if isinstance(low, int) and isinstance(high, int):
                        params[name] = np.random.randint(low, high + 1)
                    else:
                        params[name] = np.random.uniform(low, high)
                else:
                    raise ValueError(f"Unsupported distribution type for {name}")
                    
            combinations.append(params)
            
        return combinations
        
    def _calculate_parameter_importance(self, 
                                      results: List[Dict[str, Any]], 
                                      param_grid: Dict[str, List[Any]]) -> Dict[str, float]:
        """Calculate parameter importance using variance analysis"""
        
        if not results:
            return {}
            
        importance = {}
        
        for param_name in param_grid.keys():
            # Group results by parameter value
            param_groups = {}
            
            for result in results:
                value = result["parameters"][param_name]
                if value not in param_groups:
                    param_groups[value] = []
                param_groups[value].append(result["score"])
                
            # Calculate variance between groups
            if len(param_groups) > 1:
                group_means = [np.mean(scores) for scores in param_groups.values()]
                importance[param_name] = np.var(group_means)
            else:
                importance[param_name] = 0.0
                
        # Normalize
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
            
        return importance
        
    def _calculate_parameter_importance_random(self, 
                                             results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate parameter importance for random search results"""
        
        if not results:
            return {}
            
        # Extract parameter values and scores
        param_names = list(results[0]["parameters"].keys())
        importance = {}
        
        for param_name in param_names:
            values = []
            scores = []
            
            for result in results:
                try:
                    # Convert to float for correlation
                    value = float(result["parameters"][param_name])
                    values.append(value)
                    scores.append(result["score"])
                except (ValueError, TypeError):
                    # Skip non-numeric parameters
                    continue
                    
            if len(values) > 1 and len(set(values)) > 1:
                # Calculate correlation
                correlation = abs(np.corrcoef(values, scores)[0, 1])
                importance[param_name] = correlation if not np.isnan(correlation) else 0.0
            else:
                importance[param_name] = 0.0
                
        # Normalize
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
            
        return importance
        
    def _calculate_parameter_stability(self, window_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate parameter stability across windows"""
        
        if not window_results:
            return {}
            
        # Extract parameter values from each window
        param_names = list(window_results[0]["best_params"].keys())
        stability = {}
        
        for param_name in param_names:
            values = [w["best_params"][param_name] for w in window_results]
            
            # Calculate coefficient of variation
            if len(values) > 1:
                try:
                    values_float = [float(v) for v in values]
                    mean_val = np.mean(values_float)
                    std_val = np.std(values_float)
                    
                    if mean_val != 0:
                        cv = std_val / abs(mean_val)
                        stability[param_name] = 1.0 / (1.0 + cv)  # Higher is more stable
                    else:
                        stability[param_name] = 1.0 if std_val == 0 else 0.0
                        
                except (ValueError, TypeError):
                    # For non-numeric parameters, use frequency of mode
                    mode_count = max(values.count(v) for v in set(values))
                    stability[param_name] = mode_count / len(values)
            else:
                stability[param_name] = 1.0
                
        return stability
        
    def save_results(self, results: OptimizationResult, filepath: str):
        """Save optimization results to file"""
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
            
        logger.info(f"Optimization results saved to {filepath}")
        
    def load_results(self, filepath: str) -> OptimizationResult:
        """Load optimization results from file"""
        
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
            
        return results
        
    def plot_convergence(self, results: OptimizationResult):
        """Plot optimization convergence"""
        
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(results.convergence_history)
            plt.xlabel('Iteration')
            plt.ylabel(f'Best {results.optimization_metric}')
            plt.title('Optimization Convergence')
            plt.grid(True)
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            
    def plot_parameter_importance(self, results: OptimizationResult):
        """Plot parameter importance"""
        
        try:
            import matplotlib.pyplot as plt
            
            params = list(results.parameter_importance.keys())
            importance = list(results.parameter_importance.values())
            
            plt.figure(figsize=(10, 6))
            plt.bar(params, importance)
            plt.xlabel('Parameter')
            plt.ylabel('Importance')
            plt.title('Parameter Importance')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            
    def generate_report(self, results: OptimizationResult) -> str:
        """Generate text report of optimization results"""
        
        report = []
        report.append("=" * 50)
        report.append("OPTIMIZATION REPORT")
        report.append("=" * 50)
        report.append(f"Optimization Metric: {results.optimization_metric}")
        report.append(f"Total Iterations: {results.total_iterations}")
        report.append(f"Execution Time: {results.execution_time:.2f} seconds")
        report.append("")
        
        report.append("BEST PARAMETERS:")
        for param, value in results.best_params.items():
            report.append(f"  {param}: {value}")
        report.append(f"\nBest Score: {results.best_score:.4f}")
        report.append("")
        
        report.append("PARAMETER IMPORTANCE:")
        sorted_importance = sorted(results.parameter_importance.items(), 
                                 key=lambda x: x[1], reverse=True)
        for param, importance in sorted_importance:
            report.append(f"  {param}: {importance:.3f}")
        report.append("")
        
        report.append("TOP 10 RESULTS:")
        sorted_results = sorted(results.all_results, 
                              key=lambda x: x['score'], reverse=True)[:10]
        
        for i, result in enumerate(sorted_results, 1):
            report.append(f"\n{i}. Score: {result['score']:.4f}")
            for param, value in result['parameters'].items():
                report.append(f"   {param}: {value}")
                
        return "\n".join(report)


class MultiObjectiveOptimizer(StrategyOptimizer):
    """
    Multi-objective optimization for strategies
    Optimizes multiple metrics simultaneously
    """
    
    def __init__(self, objective_functions: Dict[str, Callable], n_jobs: int = -1):
        """
        Initialize multi-objective optimizer
        
        Args:
            objective_functions: Dictionary of metric name to objective function
            n_jobs: Number of parallel jobs
        """
        self.objective_functions = objective_functions
        self.n_jobs = n_jobs if n_jobs > 0 else None
        self.results_cache = {}
        
    def pareto_optimization(self,
                          param_grid: Dict[str, List[Any]],
                          weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Find Pareto optimal solutions
        
        Args:
            param_grid: Parameter grid
            weights: Optional weights for objectives (if None, find all Pareto solutions)
            
        Returns:
            Dictionary with Pareto frontier and best weighted solution
        """
        
        logger.info("Starting Pareto optimization")
        
        # Generate parameter combinations
        param_combinations = self._generate_grid_combinations(param_grid)
        
        # Evaluate all combinations
        all_results = []
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            
            for params in param_combinations:
                future = executor.submit(self._evaluate_multi_objective, params)
                futures.append((params, future))
                
            for params, future in futures:
                try:
                    scores = future.result()
                    result = {
                        "parameters": params,
                        "scores": scores
                    }
                    all_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error evaluating parameters {params}: {e}")
                    
        # Find Pareto frontier
        pareto_frontier = self._find_pareto_frontier(all_results)
        
        # Find best weighted solution if weights provided
        best_weighted = None
        if weights:
            best_weighted = self._find_best_weighted(pareto_frontier, weights)
            
        return {
            "pareto_frontier": pareto_frontier,
            "best_weighted": best_weighted,
            "all_results": all_results,
            "n_pareto_solutions": len(pareto_frontier)
        }
        
    def _evaluate_multi_objective(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate parameters for all objectives"""
        
        scores = {}
        
        for metric, func in self.objective_functions.items():
            try:
                scores[metric] = func(params, metric)
            except Exception as e:
                logger.error(f"Error evaluating {metric} for {params}: {e}")
                scores[metric] = -float('inf')
                
        return scores
        
    def _find_pareto_frontier(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find Pareto optimal solutions"""
        
        pareto_frontier = []
        
        for i, result_i in enumerate(results):
            dominated = False
            
            for j, result_j in enumerate(results):
                if i == j:
                    continue
                    
                # Check if result_j dominates result_i
                scores_i = list(result_i["scores"].values())
                scores_j = list(result_j["scores"].values())
                
                if all(s_j >= s_i for s_j, s_i in zip(scores_j, scores_i)) and \
                   any(s_j > s_i for s_j, s_i in zip(scores_j, scores_i)):
                    dominated = True
                    break
                    
            if not dominated:
                pareto_frontier.append(result_i)
                
        return pareto_frontier
        
    def _find_best_weighted(self, 
                          pareto_frontier: List[Dict[str, Any]], 
                          weights: Dict[str, float]) -> Dict[str, Any]:
        """Find best solution using weighted sum"""
        
        best_score = -float('inf')
        best_solution = None
        
        for solution in pareto_frontier:
            weighted_score = sum(
                solution["scores"][metric] * weight 
                for metric, weight in weights.items()
            )
            
            if weighted_score > best_score:
                best_score = weighted_score
                best_solution = solution
                
        return best_solution