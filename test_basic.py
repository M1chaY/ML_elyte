#!/usr/bin/env python
"""Simple test to verify ML_elyte basic functionality."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from ml_elyte import QM9Dataset, MolecularDataLoader, MolecularFeaturizer, RegressionMetrics
from ml_elyte.models import MLPRegressor, MLModel

def test_basic_functionality():
    """Test basic ML_elyte functionality."""
    print("Testing ML_elyte basic functionality...")
    
    # 1. Test QM9Dataset creation
    print("1. Creating QM9Dataset...")
    dataset = QM9Dataset(
        data_dir="./dummy_data",  # Will use dummy data
        target_properties=['mu', 'alpha'],
        use_features=['descriptors'],
        cache_features=False
    )
    print(f"   Dataset created with {len(dataset)} samples")
    
    # 2. Test data sample
    print("2. Testing data sample...")
    sample = dataset[0]
    print(f"   Sample keys: {list(sample.keys())}")
    print(f"   Feature shape: {sample['features'].shape}")
    print(f"   Target shape: {sample['targets'].shape}")
    
    # 3. Test MolecularFeaturizer
    print("3. Testing MolecularFeaturizer...")
    featurizer = MolecularFeaturizer(feature_types=['descriptors'])
    features = featurizer.featurize('CCO')
    print(f"   Feature vector shape: {features.shape}")
    print(f"   Feature dimension: {featurizer.get_feature_dim()}")
    
    # 4. Test MolecularDataLoader
    print("4. Testing MolecularDataLoader...")
    data_loader = MolecularDataLoader(
        dataset=dataset,
        batch_size=16,
        num_workers=0  # Avoid multiprocessing issues
    )
    train_loader, val_loader, test_loader = data_loader.get_all_loaders()
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Test batch
    batch = next(iter(train_loader))
    print(f"   Batch feature shape: {batch['features'].shape}")
    print(f"   Batch target shape: {batch['targets'].shape}")
    
    # 5. Test PyTorch model
    print("5. Testing PyTorch MLPRegressor...")
    input_dim = data_loader.get_feature_dim()
    output_dim = data_loader.get_target_dim()
    
    model = MLPRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=[64, 32],
        dropout_rate=0.1
    )
    
    # Test forward pass
    with torch.no_grad():
        predictions = model(batch['features'])
    print(f"   Model prediction shape: {predictions.shape}")
    print(f"   Model parameters: {model.get_num_parameters():,}")
    
    # 6. Test scikit-learn model
    print("6. Testing scikit-learn MLModel...")
    
    # Prepare data for sklearn
    X_train = []
    y_train = []
    for i, batch in enumerate(train_loader):
        X_train.append(batch['features'].numpy())
        y_train.append(batch['targets'].numpy())
        if i >= 2:  # Just use a few batches for quick test
            break
    
    X_train = np.vstack(X_train)
    y_train = np.vstack(y_train)
    
    sklearn_model = MLModel(
        model_type='ridge',
        multi_output=True,
        scale_features=True
    )
    sklearn_model.fit(X_train, y_train)
    sklearn_pred = sklearn_model.predict(X_train[:5])  # Predict on first 5 samples
    print(f"   Sklearn prediction shape: {sklearn_pred.shape}")
    
    # 7. Test metrics
    print("7. Testing RegressionMetrics...")
    metrics = RegressionMetrics(property_names=['mu', 'alpha'])
    
    # Create dummy predictions for testing
    y_true = torch.randn(10, 2)
    y_pred = y_true + torch.randn(10, 2) * 0.1  # Add small noise
    
    results = metrics.compute_metrics(y_true, y_pred, per_property=True)
    print("   Computed metrics:")
    print(f"   - Per property results available: {'per_property' in results}")
    if 'per_property' in results:
        for prop, prop_metrics in results['per_property'].items():
            print(f"     {prop}: MAE={prop_metrics['mae']:.4f}, R²={prop_metrics['r2']:.4f}")
    
    print("\n✓ All basic functionality tests passed!")
    return True

if __name__ == '__main__':
    try:
        test_basic_functionality()
        print("SUCCESS: ML_elyte is working correctly!")
    except Exception as e:
        print(f"ERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)