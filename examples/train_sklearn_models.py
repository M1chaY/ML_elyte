#!/usr/bin/env python
"""
Example script for training scikit-learn models on QM9 dataset.

This script demonstrates how to:
1. Load and preprocess QM9 data  
2. Extract molecular features
3. Train traditional ML models
4. Perform hyperparameter tuning
5. Evaluate model performance
6. Compare multiple models
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import ML_elyte modules
from ml_elyte.data import QM9Dataset
from ml_elyte.models import MLModel, get_default_param_grids
from ml_elyte.features import MolecularFeaturizer
from ml_elyte.metrics import RegressionMetrics
from ml_elyte.utils import normalize_data, save_predictions


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train scikit-learn models on QM9 dataset')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory containing QM9 data')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Validation set size')
    
    # Model arguments  
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['random_forest', 'gradient_boosting', 'ridge'],
                       choices=['random_forest', 'gradient_boosting', 'ridge', 
                               'lasso', 'elastic_net', 'svr', 'mlp'],
                       help='Types of models to train')
    parser.add_argument('--normalize_features', action='store_true',
                       help='Normalize features before training')
    parser.add_argument('--normalize_targets', action='store_true',
                       help='Normalize target values')
    
    # Feature arguments
    parser.add_argument('--feature_types', type=str, nargs='+',
                       default=['descriptors', 'fingerprints'],
                       help='Types of molecular features to use')
    parser.add_argument('--target_properties', type=str, nargs='+',
                       default=['mu', 'alpha', 'homo', 'lumo', 'gap'],
                       help='Target properties to predict')
    
    # Training arguments
    parser.add_argument('--hyperparameter_search', action='store_true',
                       help='Perform hyperparameter search')
    parser.add_argument('--search_type', type=str, default='grid',
                       choices=['grid', 'random'],
                       help='Type of hyperparameter search')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./results_sklearn',
                       help='Directory to save results')
    parser.add_argument('--save_models', action='store_true',
                       help='Save trained models')
    
    return parser.parse_args()


def prepare_data(args):
    """Prepare dataset for training."""
    print("Preparing dataset...")
    
    # Create dataset (using dummy data if real QM9 not available)
    dataset = QM9Dataset(
        data_dir=args.data_dir,
        target_properties=args.target_properties,
        use_features=args.feature_types,
        cache_features=True
    )
    
    # Extract all data
    print("Extracting features and targets...")
    features = []
    targets = []
    smiles = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        features.append(sample['features'].numpy())
        targets.append(sample['targets'].numpy())
        smiles.append(sample['smiles'])
    
    X = np.array(features)
    y = np.array(targets)
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    return X, y, smiles


def split_data(X, y, smiles, args):
    """Split data into train/val/test sets."""
    print("Splitting data...")
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test, smiles_temp, smiles_test = train_test_split(
        X, y, smiles, test_size=args.test_size, random_state=42
    )
    
    # Second split: separate train and validation
    val_size_adjusted = args.val_size / (1 - args.test_size)
    X_train, X_val, y_train, y_val, smiles_train, smiles_val = train_test_split(
        X_temp, y_temp, smiles_temp, test_size=val_size_adjusted, random_state=42
    )
    
    print(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return {
        'X_train': X_train, 'y_train': y_train, 'smiles_train': smiles_train,
        'X_val': X_val, 'y_val': y_val, 'smiles_val': smiles_val,
        'X_test': X_test, 'y_test': y_test, 'smiles_test': smiles_test
    }


def preprocess_data(data, args):
    """Preprocess features and targets."""
    print("Preprocessing data...")
    
    # Normalize features if requested
    if args.normalize_features:
        X_train_norm, feature_scaler, X_val_norm, X_test_norm = normalize_data(
            data['X_train'], data['X_val'], data['X_test'], method='standard'
        )
        data['X_train'] = X_train_norm
        data['X_val'] = X_val_norm
        data['X_test'] = X_test_norm
        data['feature_scaler'] = feature_scaler
        print("Features normalized using StandardScaler")
    
    # Normalize targets if requested
    if args.normalize_targets:
        from ml_elyte.utils.data_utils import normalize_targets
        y_train_norm, target_scaler, y_val_norm, y_test_norm = normalize_targets(
            data['y_train'], data['y_val'], data['y_test'], method='standard'
        )
        data['y_train'] = y_train_norm
        data['y_val'] = y_val_norm
        data['y_test'] = y_test_norm
        data['target_scaler'] = target_scaler
        print("Targets normalized using StandardScaler")
    
    return data


def train_model(model_type, data, args):
    """Train a single model."""
    print(f"Training {model_type} model...")
    
    # Create model
    model = MLModel(
        model_type=model_type,
        multi_output=True,
        scale_features=False  # Already scaled if requested
    )
    
    # Hyperparameter search if requested
    if args.hyperparameter_search:
        print(f"Performing hyperparameter search for {model_type}...")
        param_grids = get_default_param_grids()
        param_grid = param_grids.get(model_type, {})
        
        if param_grid:
            search_results = model.hyperparameter_search(
                data['X_train'], data['y_train'],
                param_grid=param_grid,
                search_type=args.search_type,
                cv=args.cv_folds,
                n_iter=20 if args.search_type == 'random' else None
            )
            print(f"Best parameters: {search_results['best_params']}")
            print(f"Best CV score: {search_results['best_score']:.4f}")
        else:
            print(f"No parameter grid defined for {model_type}, using default parameters")
            model.fit(data['X_train'], data['y_train'])
    else:
        # Train with default parameters
        model.fit(data['X_train'], data['y_train'])
    
    return model


def evaluate_model(model, data, model_type, args):
    """Evaluate a trained model."""
    # Make predictions
    train_pred = model.predict(data['X_train'])
    val_pred = model.predict(data['X_val'])
    test_pred = model.predict(data['X_test'])
    
    # Inverse transform if targets were normalized
    if args.normalize_targets and 'target_scaler' in data:
        from ml_elyte.utils.data_utils import inverse_transform_targets
        
        # Inverse transform targets
        y_train = inverse_transform_targets(
            data['y_train'], data['target_scaler'], 
            single_target=(data['y_train'].ndim == 1)
        )
        y_val = inverse_transform_targets(
            data['y_val'], data['target_scaler'],
            single_target=(data['y_val'].ndim == 1)
        )
        y_test = inverse_transform_targets(
            data['y_test'], data['target_scaler'],
            single_target=(data['y_test'].ndim == 1)
        )
        
        # Inverse transform predictions
        train_pred = inverse_transform_targets(
            train_pred, data['target_scaler'],
            single_target=(train_pred.ndim == 1)
        )
        val_pred = inverse_transform_targets(
            val_pred, data['target_scaler'],
            single_target=(val_pred.ndim == 1)
        )
        test_pred = inverse_transform_targets(
            test_pred, data['target_scaler'],
            single_target=(test_pred.ndim == 1)
        )
    else:
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']
    
    # Compute metrics
    metrics = RegressionMetrics(property_names=args.target_properties)
    
    train_metrics = metrics.compute_metrics(y_train, train_pred)
    val_metrics = metrics.compute_metrics(y_val, val_pred)
    test_metrics = metrics.compute_metrics(y_test, test_pred)
    
    return {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics,
        'predictions': {
            'train': train_pred,
            'val': val_pred,
            'test': test_pred
        }
    }


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare data
    X, y, smiles = prepare_data(args)
    
    # Split data
    data = split_data(X, y, smiles, args)
    
    # Preprocess data
    data = preprocess_data(data, args)
    
    # Train and evaluate models
    model_results = {}
    trained_models = {}
    
    for model_type in args.models:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()} Model")
        print(f"{'='*50}")
        
        # Train model
        model = train_model(model_type, data, args)
        trained_models[model_type] = model
        
        # Evaluate model
        results = evaluate_model(model, data, model_type, args)
        model_results[model_type] = results
        
        # Print results
        metrics_obj = RegressionMetrics(property_names=args.target_properties)
        print(f"\n{model_type.upper()} Results:")
        metrics_obj.print_metrics(results['test'], "Test Set Performance")
        
        # Save predictions
        save_predictions(
            results['predictions']['test'],
            data['smiles_test'],
            args.target_properties,
            os.path.join(args.output_dir, f'{model_type}_test_predictions.csv')
        )
        
        # Save model if requested
        if args.save_models:
            model.save_model(os.path.join(args.output_dir, f'{model_type}_model.pkl'))
    
    # Compare models
    print(f"\n{'='*50}")
    print("MODEL COMPARISON")
    print(f"{'='*50}")
    
    # Extract test metrics for comparison
    test_metrics = {model_type: results['test'] for model_type, results in model_results.items()}
    
    # Create comparison plots
    metrics_obj = RegressionMetrics(property_names=args.target_properties)
    
    for metric_name in ['mae', 'rmse', 'r2']:
        metrics_obj.plot_metrics_comparison(
            test_metrics, metric_name,
            title=f'{metric_name.upper()} Comparison Across Models',
            save_path=os.path.join(args.output_dir, f'comparison_{metric_name}.png')
        )
        plt.close()
    
    # Create summary table
    summary_data = []
    for model_type, results in model_results.items():
        if 'average' in results['test']:
            summary_data.append({
                'Model': model_type,
                'MAE': results['test']['average']['mae'],
                'RMSE': results['test']['average']['rmse'],
                'R²': results['test']['average']['r2']
            })
        else:
            summary_data.append({
                'Model': model_type,
                'MAE': results['test']['mae'],
                'RMSE': results['test']['rmse'],
                'R²': results['test']['r2']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('R²', ascending=False)
    
    print("\nModel Performance Summary (Test Set):")
    print(summary_df.to_string(index=False, float_format='%.4f'))
    
    # Save summary
    summary_df.to_csv(os.path.join(args.output_dir, 'model_comparison.csv'), index=False)
    
    print(f"\nResults saved to: {args.output_dir}")
    print("Training completed!")


if __name__ == '__main__':
    main()