"""Regression metrics for molecular property prediction."""

import numpy as np
import torch
from typing import Dict, List, Optional, Union, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


class RegressionMetrics:
    """
    Comprehensive regression metrics for molecular property prediction.
    
    Computes MAE, RMSE, R² for each property and provides visualization utilities.
    """
    
    def __init__(self, property_names: Optional[List[str]] = None):
        """
        Initialize RegressionMetrics.
        
        Args:
            property_names: Names of target properties
        """
        self.property_names = property_names
    
    def compute_metrics(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        per_property: bool = True
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Compute regression metrics.
        
        Args:
            y_true: True target values (n_samples, n_properties)
            y_pred: Predicted values (n_samples, n_properties)
            per_property: Whether to compute metrics per property
            
        Returns:
            Dictionary of computed metrics
        """
        # Convert to numpy if needed
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        
        # Ensure 2D arrays
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        n_properties = y_true.shape[1]
        
        # Set property names if not provided
        if self.property_names is None:
            self.property_names = [f'property_{i}' for i in range(n_properties)]
        
        metrics = {}
        
        if per_property and n_properties > 1:
            # Compute metrics for each property
            metrics['per_property'] = {}
            
            for i, prop_name in enumerate(self.property_names):
                prop_metrics = self._compute_single_property_metrics(
                    y_true[:, i], y_pred[:, i]
                )
                metrics['per_property'][prop_name] = prop_metrics
            
            # Compute average metrics across properties
            metrics['average'] = self._average_metrics(metrics['per_property'])
        
        else:
            # Compute overall metrics
            if n_properties == 1:
                metrics = self._compute_single_property_metrics(
                    y_true.flatten(), y_pred.flatten()
                )
            else:
                # Multi-output metrics
                metrics['mae'] = mean_absolute_error(y_true, y_pred, multioutput='uniform_average')
                metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='uniform_average'))
                metrics['r2'] = r2_score(y_true, y_pred, multioutput='uniform_average')
        
        return metrics
    
    def _compute_single_property_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute metrics for a single property."""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        mape = self._compute_mape(y_true, y_pred)
        max_error = np.max(np.abs(y_true - y_pred))
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape),
            'max_error': float(max_error)
        }
    
    def _compute_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Absolute Percentage Error."""
        mask = y_true != 0
        if not np.any(mask):
            return float('inf')
        
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def _average_metrics(self, property_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Average metrics across properties."""
        metric_names = list(next(iter(property_metrics.values())).keys())
        avg_metrics = {}
        
        for metric_name in metric_names:
            values = [prop_metrics[metric_name] for prop_metrics in property_metrics.values()]
            avg_metrics[metric_name] = float(np.mean(values))
        
        return avg_metrics
    
    def print_metrics(
        self,
        metrics: Dict,
        title: str = "Regression Metrics"
    ):
        """Print metrics in a formatted way."""
        print(f"\n{title}")
        print("=" * len(title))
        
        if 'per_property' in metrics:
            # Print per-property metrics
            for prop_name, prop_metrics in metrics['per_property'].items():
                print(f"\n{prop_name}:")
                for metric_name, value in prop_metrics.items():
                    if metric_name == 'mape' and value == float('inf'):
                        print(f"  {metric_name.upper()}: inf (division by zero)")
                    else:
                        print(f"  {metric_name.upper()}: {value:.4f}")
            
            # Print average metrics
            print(f"\nAverage across properties:")
            for metric_name, value in metrics['average'].items():
                if metric_name == 'mape' and value == float('inf'):
                    print(f"  {metric_name.upper()}: inf")
                else:
                    print(f"  {metric_name.upper()}: {value:.4f}")
        
        else:
            # Print single set of metrics
            for metric_name, value in metrics.items():
                if metric_name == 'mape' and value == float('inf'):
                    print(f"  {metric_name.upper()}: inf")
                else:
                    print(f"  {metric_name.upper()}: {value:.4f}")
    
    def plot_predictions(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        property_idx: int = 0,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """
        Plot predicted vs true values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            property_idx: Index of property to plot (for multi-target)
            title: Plot title
            save_path: Path to save plot
            figsize: Figure size
        """
        # Convert to numpy if needed
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        
        # Handle dimensions
        if y_true.ndim > 1:
            y_true = y_true[:, property_idx]
            y_pred = y_pred[:, property_idx]
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        # Labels and title
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        
        if title is None:
            if self.property_names and property_idx < len(self.property_names):
                title = f'Predictions vs True Values: {self.property_names[property_idx]}'
            else:
                title = 'Predictions vs True Values'
        plt.title(title)
        
        # Add R² to plot
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_residuals(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        property_idx: int = 0,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """
        Plot residuals vs predicted values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            property_idx: Index of property to plot
            title: Plot title
            save_path: Path to save plot
            figsize: Figure size
        """
        # Convert to numpy if needed
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        
        # Handle dimensions
        if y_true.ndim > 1:
            y_true = y_true[:, property_idx]
            y_pred = y_pred[:, property_idx]
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Scatter plot
        plt.scatter(y_pred, residuals, alpha=0.6, s=20)
        
        # Zero line
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        
        # Labels and title
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        
        if title is None:
            if self.property_names and property_idx < len(self.property_names):
                title = f'Residuals vs Predictions: {self.property_names[property_idx]}'
            else:
                title = 'Residuals vs Predictions'
        plt.title(title)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict],
        metric_name: str = 'mae',
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot comparison of metrics across different models or conditions.
        
        Args:
            metrics_dict: Dict mapping model names to metric dictionaries
            metric_name: Metric to plot ('mae', 'rmse', 'r2')
            title: Plot title
            save_path: Path to save plot
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        model_names = list(metrics_dict.keys())
        
        # Check if metrics have per_property structure
        first_metrics = next(iter(metrics_dict.values()))
        if 'per_property' in first_metrics:
            # Plot per-property metrics
            property_names = list(first_metrics['per_property'].keys())
            x = np.arange(len(property_names))
            width = 0.8 / len(model_names)
            
            for i, (model_name, metrics) in enumerate(metrics_dict.items()):
                values = [metrics['per_property'][prop][metric_name] for prop in property_names]
                plt.bar(x + i * width, values, width, label=model_name, alpha=0.8)
            
            plt.xlabel('Properties')
            plt.xticks(x + width * (len(model_names) - 1) / 2, property_names, rotation=45)
        
        else:
            # Plot overall metrics
            values = [metrics[metric_name] for metrics in metrics_dict.values()]
            plt.bar(model_names, values, alpha=0.8)
            plt.xlabel('Models')
            plt.xticks(rotation=45)
        
        plt.ylabel(metric_name.upper())
        
        if title is None:
            title = f'{metric_name.upper()} Comparison'
        plt.title(title)
        
        if 'per_property' in first_metrics:
            plt.legend()
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()