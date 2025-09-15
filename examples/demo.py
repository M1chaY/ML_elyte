#!/usr/bin/env python
"""
Simple demo script showing how to use ML_elyte for molecular property prediction.

This script demonstrates the basic workflow:
1. Create a dataset with dummy QM9 data
2. Extract molecular features
3. Train both PyTorch and scikit-learn models  
4. Evaluate and compare performance
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add parent directory to path to import ml_elyte
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_elyte import QM9Dataset, MolecularDataLoader, MolecularFeaturizer, RegressionMetrics
from ml_elyte.models import MLPRegressor, MLModel


def main():
    """Run the demo."""
    print("ML_elyte Demo: Molecular Property Prediction")
    print("=" * 50)
    
    # Create output directory
    output_dir = "./demo_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Create dataset with dummy data
    print("1. Creating QM9 dataset...")
    dataset = QM9Dataset(
        data_dir="./dummy_data",  # Will create dummy data since path doesn't exist
        target_properties=['mu', 'alpha', 'homo', 'lumo'],
        use_features=['descriptors', 'fingerprints'],
        cache_features=True
    )
    print(f"Dataset created with {len(dataset)} molecules")
    
    # 2. Create data loaders for PyTorch
    print("\n2. Creating data loaders...")
    data_loader = MolecularDataLoader(
        dataset=dataset,
        batch_size=32,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    train_loader, val_loader, test_loader = data_loader.get_all_loaders()
    feature_dim = data_loader.get_feature_dim()
    target_dim = data_loader.get_target_dim()
    
    print(f"Feature dimension: {feature_dim}")
    print(f"Target dimension: {target_dim}")
    
    # 3. Train PyTorch model
    print("\n3. Training PyTorch MLP model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    pytorch_model = MLPRegressor(
        input_dim=feature_dim,
        output_dim=target_dim,
        hidden_dims=[256, 128, 64],
        dropout_rate=0.2
    ).to(device)
    
    # Simple training loop
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=0.001)
    
    # Train for a few epochs
    pytorch_model.train()
    for epoch in range(10):
        total_loss = 0
        for batch in train_loader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            
            optimizer.zero_grad()
            predictions = pytorch_model(features)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")
    
    # 4. Evaluate PyTorch model
    print("\n4. Evaluating PyTorch model...")
    pytorch_model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            predictions = pytorch_model(features)
            test_predictions.append(predictions.cpu())
            test_targets.append(targets.cpu())
    
    pytorch_pred = torch.cat(test_predictions, dim=0)
    pytorch_true = torch.cat(test_targets, dim=0)
    
    # 5. Train scikit-learn models
    print("\n5. Training scikit-learn models...")
    
    # Extract features for sklearn
    X_train, y_train = [], []
    for batch in train_loader:
        X_train.append(batch['features'].numpy())
        y_train.append(batch['targets'].numpy())
    
    X_train = np.vstack(X_train)
    y_train = np.vstack(y_train)
    
    X_test, y_test = [], []
    for batch in test_loader:
        X_test.append(batch['features'].numpy())
        y_test.append(batch['targets'].numpy())
    
    X_test = np.vstack(X_test)
    y_test = np.vstack(y_test)
    
    # Train Random Forest
    rf_model = MLModel(model_type='random_forest', multi_output=True, scale_features=True)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    # Train Ridge Regression
    ridge_model = MLModel(model_type='ridge', multi_output=True, scale_features=True)
    ridge_model.fit(X_train, y_train)
    ridge_pred = ridge_model.predict(X_test)
    
    # 6. Compare model performance
    print("\n6. Comparing model performance...")
    metrics = RegressionMetrics(property_names=['mu', 'alpha', 'homo', 'lumo'])
    
    pytorch_metrics = metrics.compute_metrics(pytorch_true, pytorch_pred)
    rf_metrics = metrics.compute_metrics(y_test, rf_pred)
    ridge_metrics = metrics.compute_metrics(y_test, ridge_pred)
    
    print("\nPyTorch MLP Results:")
    metrics.print_metrics(pytorch_metrics, "PyTorch MLP")
    
    print("\nRandom Forest Results:")
    metrics.print_metrics(rf_metrics, "Random Forest")
    
    print("\nRidge Regression Results:")
    metrics.print_metrics(ridge_metrics, "Ridge Regression")
    
    # 7. Create visualizations
    print("\n7. Creating visualizations...")
    
    # Plot predictions for first property (mu)
    metrics.plot_predictions(
        pytorch_true, pytorch_pred,
        property_idx=0,
        title='PyTorch MLP: Dipole Moment Predictions',
        save_path=os.path.join(output_dir, 'pytorch_mu_predictions.png')
    )
    plt.close()
    
    metrics.plot_predictions(
        y_test, rf_pred,
        property_idx=0,
        title='Random Forest: Dipole Moment Predictions', 
        save_path=os.path.join(output_dir, 'rf_mu_predictions.png')
    )
    plt.close()
    
    # Compare models
    all_metrics = {
        'PyTorch_MLP': pytorch_metrics,
        'Random_Forest': rf_metrics,
        'Ridge_Regression': ridge_metrics
    }
    
    metrics.plot_metrics_comparison(
        all_metrics, 'mae',
        title='Mean Absolute Error Comparison',
        save_path=os.path.join(output_dir, 'mae_comparison.png')
    )
    plt.close()
    
    metrics.plot_metrics_comparison(
        all_metrics, 'r2',
        title='R² Score Comparison',
        save_path=os.path.join(output_dir, 'r2_comparison.png')
    )
    plt.close()
    
    # 8. Feature importance (for Random Forest)
    print("\n8. Analyzing feature importance...")
    feature_importance = rf_model.feature_importance()
    if feature_importance is not None:
        # Get feature names
        featurizer = MolecularFeaturizer(feature_types=['descriptors', 'fingerprints'])
        feature_names = featurizer.get_feature_names()
        
        # Plot top 20 most important features
        top_indices = np.argsort(feature_importance)[-20:]
        top_importance = feature_importance[top_indices]
        top_names = [feature_names[i] if i < len(feature_names) else f'feature_{i}' 
                    for i in top_indices]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_importance)), top_importance)
        plt.yticks(range(len(top_importance)), top_names)
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Most Important Features (Random Forest)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Feature importance plot saved")
    
    print(f"\n✓ Demo completed! Results saved to: {output_dir}")
    print("\nFiles created:")
    for file in os.listdir(output_dir):
        print(f"  - {file}")


if __name__ == '__main__':
    main()