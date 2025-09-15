"""Scikit-learn model implementations."""

import numpy as np
from typing import List, Optional, Dict, Any, Union, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor as SKMLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib


class MLModel:
    """
    Wrapper class for scikit-learn models for molecular property prediction.
    Supports multiple regression algorithms and multi-target prediction.
    """
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        multi_output: bool = True,
        scale_features: bool = True,
        **model_kwargs
    ):
        """
        Initialize MLModel.
        
        Args:
            model_type: Type of sklearn model to use
            multi_output: Whether to use MultiOutputRegressor for multi-target prediction
            scale_features: Whether to scale features before training
            **model_kwargs: Additional arguments for the base model
        """
        self.model_type = model_type
        self.multi_output = multi_output
        self.scale_features = scale_features
        self.model_kwargs = model_kwargs
        
        # Initialize base model
        self.base_model = self._get_base_model()
        
        # Wrap in MultiOutputRegressor if needed
        if multi_output:
            self.model = MultiOutputRegressor(self.base_model)
        else:
            self.model = self.base_model
        
        # Initialize scaler
        self.scaler = StandardScaler() if scale_features else None
        
        # Track if model is trained
        self.is_trained = False
    
    def _get_base_model(self):
        """Get the base sklearn model."""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
                **self.model_kwargs
            )
        
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                **self.model_kwargs
            )
        
        elif self.model_type == 'ridge':
            return Ridge(
                alpha=1.0,
                random_state=42,
                **self.model_kwargs
            )
        
        elif self.model_type == 'lasso':
            return Lasso(
                alpha=1.0,
                max_iter=1000,
                random_state=42,
                **self.model_kwargs
            )
        
        elif self.model_type == 'elastic_net':
            return ElasticNet(
                alpha=1.0,
                l1_ratio=0.5,
                max_iter=1000,
                random_state=42,
                **self.model_kwargs
            )
        
        elif self.model_type == 'svr':
            return SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                **self.model_kwargs
            )
        
        elif self.model_type == 'mlp':
            return SKMLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='constant',
                learning_rate_init=0.001,
                max_iter=200,
                random_state=42,
                **self.model_kwargs
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target matrix (n_samples, n_targets)
        """
        # Scale features if requested
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predictions (n_samples, n_targets)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale features if scaler was used during training
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Get model score (R²).
        
        Args:
            X: Feature matrix
            y: True targets
            
        Returns:
            R² score
        """
        predictions = self.predict(X)
        return r2_score(y, predictions, multioutput='uniform_average')
    
    def feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importances if available.
        
        Returns:
            Feature importances or None if not available
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Extract base model for importance
        if self.multi_output:
            base_model = self.model.estimators_[0]  # First estimator
        else:
            base_model = self.model
        
        if hasattr(base_model, 'feature_importances_'):
            return base_model.feature_importances_
        elif hasattr(base_model, 'coef_'):
            return np.abs(base_model.coef_).mean(axis=0) if base_model.coef_.ndim > 1 else np.abs(base_model.coef_)
        else:
            return None
    
    def hyperparameter_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Dict[str, List],
        search_type: str = 'grid',
        cv: int = 5,
        n_iter: int = 20,
        scoring: str = 'r2',
        n_jobs: int = -1
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter search.
        
        Args:
            X: Feature matrix
            y: Target matrix
            param_grid: Parameter grid for search
            search_type: Type of search ('grid' or 'random')
            cv: Number of CV folds
            n_iter: Number of iterations for random search
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary with best parameters and score
        """
        # Scale features if requested
        X_scaled = self.scaler.fit_transform(X) if self.scaler else X
        
        # Choose search method
        if search_type == 'grid':
            search = GridSearchCV(
                self.model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs
            )
        elif search_type == 'random':
            search = RandomizedSearchCV(
                self.model, param_grid, n_iter=n_iter, cv=cv, 
                scoring=scoring, n_jobs=n_jobs, random_state=42
            )
        else:
            raise ValueError(f"Unknown search type: {search_type}")
        
        # Perform search
        search.fit(X_scaled, y)
        
        # Update model with best parameters
        self.model = search.best_estimator_
        self.is_trained = True
        
        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_
        }
    
    def save_model(self, filepath: str):
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'multi_output': self.multi_output,
            'scale_features': self.scale_features,
            'model_kwargs': self.model_kwargs
        }
        
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'MLModel':
        """Load trained model from disk."""
        model_data = joblib.load(filepath)
        
        # Create new instance
        instance = cls(
            model_type=model_data['model_type'],
            multi_output=model_data['multi_output'],
            scale_features=model_data['scale_features'],
            **model_data['model_kwargs']
        )
        
        # Load trained components
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.is_trained = True
        
        return instance
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        info = {
            'model_type': self.model_type,
            'multi_output': self.multi_output,
            'scale_features': self.scale_features,
            'is_trained': self.is_trained,
            'model_kwargs': self.model_kwargs
        }
        
        if self.is_trained:
            info['feature_importance_available'] = self.feature_importance() is not None
            
        return info


def get_default_param_grids() -> Dict[str, Dict]:
    """
    Get default parameter grids for hyperparameter search.
    
    Returns:
        Dictionary mapping model types to parameter grids
    """
    return {
        'random_forest': {
            'estimator__n_estimators': [50, 100, 200],
            'estimator__max_depth': [None, 10, 20, 30],
            'estimator__min_samples_split': [2, 5, 10],
            'estimator__min_samples_leaf': [1, 2, 4]
        },
        'gradient_boosting': {
            'estimator__n_estimators': [50, 100, 200],
            'estimator__learning_rate': [0.01, 0.1, 0.2],
            'estimator__max_depth': [3, 5, 7],
            'estimator__min_samples_split': [2, 5, 10]
        },
        'ridge': {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        },
        'lasso': {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        },
        'elastic_net': {
            'alpha': [0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.5, 0.9]
        },
        'svr': {
            'estimator__C': [0.1, 1.0, 10.0],
            'estimator__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'estimator__kernel': ['rbf', 'poly', 'sigmoid']
        },
        'mlp': {
            'estimator__hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100)],
            'estimator__activation': ['relu', 'tanh'],
            'estimator__alpha': [0.0001, 0.001, 0.01],
            'estimator__learning_rate': ['constant', 'adaptive']
        }
    }