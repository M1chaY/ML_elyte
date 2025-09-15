"""Data processing utilities."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split


def train_test_split_molecular(
    features: np.ndarray,
    targets: np.ndarray,
    smiles: list,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: Optional[np.ndarray] = None
) -> Dict[str, Union[np.ndarray, list]]:
    """
    Split molecular data into train/validation/test sets.
    
    Args:
        features: Feature matrix
        targets: Target matrix
        smiles: List of SMILES strings
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed
        stratify: Stratification array (optional)
        
    Returns:
        Dictionary with train/val/test splits
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test, smiles_temp, smiles_test = train_test_split(
        features, targets, smiles, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify
    )
    
    # Second split: separate train and validation from remaining data
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, smiles_train, smiles_val = train_test_split(
        X_temp, y_temp, smiles_temp,
        test_size=val_size_adjusted,
        random_state=random_state
    )
    
    return {
        'X_train': X_train, 'y_train': y_train, 'smiles_train': smiles_train,
        'X_val': X_val, 'y_val': y_val, 'smiles_val': smiles_val,
        'X_test': X_test, 'y_test': y_test, 'smiles_test': smiles_test
    }


def normalize_data(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    method: str = 'standard',
    feature_range: Tuple[float, float] = (0, 1)
) -> Tuple[np.ndarray, ...]:
    """
    Normalize features using different scaling methods.
    
    Args:
        X_train: Training features
        X_val: Validation features (optional)
        X_test: Test features (optional)
        method: Normalization method ('standard', 'minmax', 'robust')
        feature_range: Range for MinMaxScaler
        
    Returns:
        Tuple of normalized arrays and fitted scaler
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler(feature_range=feature_range)
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Fit scaler on training data
    X_train_scaled = scaler.fit_transform(X_train)
    
    results = [X_train_scaled, scaler]
    
    # Transform validation and test data if provided
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        results.append(X_val_scaled)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        results.append(X_test_scaled)
    
    return tuple(results)


def normalize_targets(
    y_train: np.ndarray,
    y_val: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    method: str = 'standard'
) -> Tuple[np.ndarray, ...]:
    """
    Normalize target values.
    
    Args:
        y_train: Training targets
        y_val: Validation targets (optional)
        y_test: Test targets (optional)
        method: Normalization method
        
    Returns:
        Tuple of normalized arrays and fitted scalers
    """
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
        single_target = True
    else:
        single_target = False
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Fit and transform training targets
    y_train_scaled = scaler.fit_transform(y_train)
    
    if single_target:
        y_train_scaled = y_train_scaled.flatten()
    
    results = [y_train_scaled, scaler]
    
    # Transform validation and test targets
    if y_val is not None:
        if y_val.ndim == 1:
            y_val = y_val.reshape(-1, 1)
        y_val_scaled = scaler.transform(y_val)
        if single_target:
            y_val_scaled = y_val_scaled.flatten()
        results.append(y_val_scaled)
    
    if y_test is not None:
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)
        y_test_scaled = scaler.transform(y_test)
        if single_target:
            y_test_scaled = y_test_scaled.flatten()
        results.append(y_test_scaled)
    
    return tuple(results)


def inverse_transform_targets(
    y_scaled: np.ndarray,
    scaler,
    single_target: bool = False
) -> np.ndarray:
    """
    Inverse transform normalized targets back to original scale.
    
    Args:
        y_scaled: Scaled target values
        scaler: Fitted scaler object
        single_target: Whether targets are single-dimensional
        
    Returns:
        Targets in original scale
    """
    if single_target and y_scaled.ndim == 1:
        y_scaled = y_scaled.reshape(-1, 1)
    
    y_original = scaler.inverse_transform(y_scaled)
    
    if single_target:
        y_original = y_original.flatten()
    
    return y_original


def remove_outliers(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'iqr',
    threshold: float = 1.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove outliers from dataset.
    
    Args:
        X: Feature matrix
        y: Target array
        method: Outlier detection method ('iqr', 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        Tuple of (X_clean, y_clean, outlier_mask)
    """
    if method == 'iqr':
        outlier_mask = _detect_outliers_iqr(y, threshold)
    elif method == 'zscore':
        outlier_mask = _detect_outliers_zscore(y, threshold)
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    # Remove outliers
    X_clean = X[~outlier_mask]
    y_clean = y[~outlier_mask]
    
    print(f"Removed {np.sum(outlier_mask)} outliers out of {len(y)} samples")
    
    return X_clean, y_clean, outlier_mask


def _detect_outliers_iqr(y: np.ndarray, threshold: float = 1.5) -> np.ndarray:
    """Detect outliers using Interquartile Range method."""
    if y.ndim > 1:
        # For multi-target, detect outliers in any target
        outliers = np.zeros(len(y), dtype=bool)
        for i in range(y.shape[1]):
            Q1 = np.percentile(y[:, i], 25)
            Q3 = np.percentile(y[:, i], 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers |= (y[:, i] < lower_bound) | (y[:, i] > upper_bound)
        return outliers
    else:
        Q1 = np.percentile(y, 25)
        Q3 = np.percentile(y, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (y < lower_bound) | (y > upper_bound)


def _detect_outliers_zscore(y: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Detect outliers using Z-score method."""
    if y.ndim > 1:
        # For multi-target, detect outliers in any target
        outliers = np.zeros(len(y), dtype=bool)
        for i in range(y.shape[1]):
            z_scores = np.abs((y[:, i] - np.mean(y[:, i])) / np.std(y[:, i]))
            outliers |= z_scores > threshold
        return outliers
    else:
        z_scores = np.abs((y - np.mean(y)) / np.std(y))
        return z_scores > threshold


def balance_dataset(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'undersample',
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance dataset for classification tasks.
    Note: This is primarily for classification, but included for completeness.
    
    Args:
        X: Feature matrix
        y: Target array (should be discrete for balancing)
        method: Balancing method ('undersample', 'oversample')
        random_state: Random seed
        
    Returns:
        Balanced X and y arrays
    """
    # This would typically use imbalanced-learn library
    # For now, just return original data
    print("Dataset balancing not implemented for regression tasks")
    return X, y


def get_data_info(X: np.ndarray, y: np.ndarray) -> Dict:
    """
    Get information about the dataset.
    
    Args:
        X: Feature matrix
        y: Target array
        
    Returns:
        Dictionary with dataset information
    """
    info = {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_targets': y.shape[1] if y.ndim > 1 else 1,
        'feature_stats': {
            'mean': np.mean(X, axis=0),
            'std': np.std(X, axis=0),
            'min': np.min(X, axis=0),
            'max': np.max(X, axis=0)
        },
        'target_stats': {}
    }
    
    if y.ndim > 1:
        for i in range(y.shape[1]):
            info['target_stats'][f'target_{i}'] = {
                'mean': float(np.mean(y[:, i])),
                'std': float(np.std(y[:, i])),
                'min': float(np.min(y[:, i])),
                'max': float(np.max(y[:, i]))
            }
    else:
        info['target_stats']['target'] = {
            'mean': float(np.mean(y)),
            'std': float(np.std(y)),
            'min': float(np.min(y)),
            'max': float(np.max(y))
        }
    
    return info