#!/usr/bin/env python
"""
Example script for training PyTorch models on QM9 dataset.

This script demonstrates how to:
1. Load and preprocess QM9 data
2. Create DataLoaders for batching
3. Train PyTorch models
4. Evaluate model performance
5. Save trained models and predictions
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import ML_elyte modules
from ml_elyte.data import QM9Dataset, MolecularDataLoader
from ml_elyte.models import get_model
from ml_elyte.features import MolecularFeaturizer
from ml_elyte.metrics import RegressionMetrics
from ml_elyte.utils import save_predictions


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train PyTorch models on QM9 dataset')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data', 
                       help='Directory containing QM9 data')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='mlp',
                       choices=['mlp', 'gnn', 'attention'],
                       help='Type of model to train')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256, 128],
                       help='Hidden layer dimensions for MLP')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--activation', type=str, default='relu',
                       help='Activation function')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for optimizer')
    parser.add_argument('--patience', type=int, default=10,
                       help='Patience for early stopping')
    
    # Feature arguments
    parser.add_argument('--feature_types', type=str, nargs='+', 
                       default=['descriptors', 'fingerprints'],
                       help='Types of molecular features to use')
    parser.add_argument('--target_properties', type=str, nargs='+',
                       default=['mu', 'alpha', 'homo', 'lumo', 'gap'],
                       help='Target properties to predict')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--save_model', action='store_true',
                       help='Save trained model')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, auto)')
    
    return parser.parse_args()


def setup_device(device_str: str):
    """Setup device for training."""
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(device).total_memory // 1024**3} GB")
    
    return device


def create_dataset_and_loaders(args):
    """Create dataset and data loaders."""
    print("Creating dataset and data loaders...")
    
    # Create dataset
    dataset = QM9Dataset(
        data_dir=args.data_dir,
        target_properties=args.target_properties,
        use_features=args.feature_types,
        cache_features=True
    )
    
    # Create data loader
    data_loader = MolecularDataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Get data loaders
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    test_loader = data_loader.get_test_loader()
    
    print(f"Dataset statistics:")
    stats = data_loader.get_dataset_stats()
    for key, value in stats.items():
        if key != 'property_stats':
            print(f"  {key}: {value}")
    
    return train_loader, val_loader, test_loader, data_loader


def create_model(args, input_dim, output_dim, device):
    """Create and initialize model."""
    print(f"Creating {args.model_type} model...")
    
    model_kwargs = {}
    if args.model_type == 'mlp':
        model_kwargs = {
            'hidden_dims': args.hidden_dims,
            'dropout_rate': args.dropout_rate,
            'activation': args.activation,
            'batch_norm': True
        }
    elif args.model_type == 'gnn':
        model_kwargs = {
            'hidden_dim': args.hidden_dims[0],
            'num_layers': 3,
            'dropout_rate': args.dropout_rate
        }
    elif args.model_type == 'attention':
        model_kwargs = {
            'hidden_dim': args.hidden_dims[0],
            'num_heads': 4,
            'num_layers': 3,
            'dropout_rate': args.dropout_rate
        }
    
    model = get_model(args.model_type, input_dim, output_dim, **model_kwargs)
    model = model.to(device)
    
    print(f"Model created with {model.get_num_parameters():,} parameters")
    return model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        features = batch['features'].to(device)
        targets = batch['targets'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate_model(model, data_loader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            
            predictions = model(features)
            loss = criterion(predictions, targets)
            
            total_loss += loss.item()
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate all predictions and targets
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    avg_loss = total_loss / len(data_loader)
    
    return avg_loss, predictions, targets


def train_model(model, train_loader, val_loader, args, device):
    """Train the model with early stopping."""
    print("Starting training...")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, 
                          weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, _, _ = evaluate_model(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 
                      os.path.join(args.output_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))
    
    return train_losses, val_losses


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = setup_device(args.device)
    
    # Create dataset and data loaders
    train_loader, val_loader, test_loader, data_loader = create_dataset_and_loaders(args)
    
    # Get dimensions
    input_dim = data_loader.get_feature_dim()
    output_dim = data_loader.get_target_dim()
    
    # Create model
    model = create_model(args, input_dim, output_dim, device)
    
    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, args, device)
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'training_history.png'))
    plt.close()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    criterion = nn.MSELoss()
    test_loss, test_predictions, test_targets = evaluate_model(
        model, test_loader, criterion, device
    )
    print(f"Test Loss: {test_loss:.4f}")
    
    # Compute detailed metrics
    metrics = RegressionMetrics(property_names=args.target_properties)
    test_metrics = metrics.compute_metrics(test_targets, test_predictions)
    metrics.print_metrics(test_metrics, "Test Set Performance")
    
    # Plot predictions
    for i, prop_name in enumerate(args.target_properties):
        metrics.plot_predictions(
            test_targets, test_predictions, 
            property_idx=i,
            title=f'Test Predictions: {prop_name}',
            save_path=os.path.join(args.output_dir, f'predictions_{prop_name}.png')
        )
        plt.close()
    
    # Save predictions
    save_predictions(
        test_predictions.numpy(),
        [],  # SMILES not available in this context
        args.target_properties,
        os.path.join(args.output_dir, 'test_predictions.csv'),
        include_smiles=False
    )
    
    # Save model if requested
    if args.save_model:
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': args.model_type,
            'input_dim': input_dim,
            'output_dim': output_dim,
            'target_properties': args.target_properties,
            'feature_types': args.feature_types
        }, os.path.join(args.output_dir, 'final_model.pth'))
        print(f"Model saved to {os.path.join(args.output_dir, 'final_model.pth')}")
    
    print("Training completed!")


if __name__ == '__main__':
    main()