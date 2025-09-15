"""DataLoader utilities for molecular datasets."""

import torch
from torch.utils.data import DataLoader, random_split
from typing import Dict, List, Optional, Tuple
from .qm9_dataset import QM9Dataset


class MolecularDataLoader:
    """
    DataLoader wrapper for molecular datasets with proper batching and splitting.
    """
    
    def __init__(
        self,
        dataset: QM9Dataset,
        batch_size: int = 32,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        """
        Initialize MolecularDataLoader.
        
        Args:
            dataset: QM9Dataset instance
            batch_size: Batch size for data loading
            train_split: Fraction of data for training
            val_split: Fraction of data for validation  
            test_split: Fraction of data for testing
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Validate splits
        if not abs(train_split + val_split + test_split - 1.0) < 1e-6:
            raise ValueError("Train, validation, and test splits must sum to 1.0")
        
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        # Create data splits
        self._create_splits()
    
    def _create_splits(self):
        """Create train/val/test splits from the dataset."""
        dataset_size = len(self.dataset)
        
        # Calculate split sizes
        train_size = int(self.train_split * dataset_size)
        val_size = int(self.val_split * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        # Create random splits
        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size], generator=generator
        )
        
        print(f"Dataset split - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    def get_train_loader(self) -> DataLoader:
        """Get training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )
    
    def get_val_loader(self) -> DataLoader:
        """Get validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )
    
    def get_test_loader(self) -> DataLoader:
        """Get test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )
    
    def get_all_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get all DataLoaders (train, val, test)."""
        return (
            self.get_train_loader(),
            self.get_val_loader(),
            self.get_test_loader()
        )
    
    @staticmethod
    def _collate_fn(batch: List[Dict]) -> Dict:
        """
        Custom collate function for batching molecular data.
        
        Args:
            batch: List of samples from dataset
            
        Returns:
            Batched data dictionary
        """
        # Stack features and targets
        features = torch.stack([sample['features'] for sample in batch])
        targets = torch.stack([sample['targets'] for sample in batch])
        
        # Keep SMILES and mol_ids as lists
        smiles = [sample['smiles'] for sample in batch]
        mol_ids = [sample['mol_id'] for sample in batch]
        
        # Property names should be the same for all samples
        property_names = batch[0]['property_names']
        
        return {
            'features': features,
            'targets': targets,
            'smiles': smiles,
            'mol_ids': mol_ids,
            'property_names': property_names
        }
    
    def get_feature_dim(self) -> int:
        """Get the dimension of molecular features."""
        sample = self.dataset[0]
        return sample['features'].shape[0]
    
    def get_target_dim(self) -> int:
        """Get the dimension of target properties."""
        sample = self.dataset[0]
        return sample['targets'].shape[0]
    
    def get_dataset_stats(self) -> Dict:
        """Get statistics about the dataset."""
        return {
            'total_size': len(self.dataset),
            'train_size': len(self.train_dataset),
            'val_size': len(self.val_dataset),
            'test_size': len(self.test_dataset),
            'feature_dim': self.get_feature_dim(),
            'target_dim': self.get_target_dim(),
            'batch_size': self.batch_size,
            'property_names': self.dataset.target_properties,
            'property_stats': self.dataset.get_property_stats()
        }