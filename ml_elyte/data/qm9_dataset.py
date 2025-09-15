"""QM9 Dataset implementation for PyTorch."""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union
from ..utils.molecular_utils import load_qm9_data, parse_smiles


class QM9Dataset(Dataset):
    """
    PyTorch Dataset for QM9 molecular property prediction.
    
    The QM9 dataset contains ~134k small organic molecules with up to 9 heavy atoms
    (C, N, O, F) along with their ground-state properties computed using DFT.
    
    Properties include:
    - mu: Dipole moment (Debye)
    - alpha: Isotropic polarizability (Bohr^3)
    - homo: Highest occupied molecular orbital energy (Hartree)
    - lumo: Lowest unoccupied molecular orbital energy (Hartree)
    - gap: HOMO-LUMO gap (Hartree)
    - r2: Electronic spatial extent (Bohr^2)
    - zpve: Zero point vibrational energy (Hartree)
    - u0: Internal energy at 0K (Hartree)
    - u298: Internal energy at 298.15K (Hartree)
    - h298: Enthalpy at 298.15K (Hartree)
    - g298: Free energy at 298.15K (Hartree)
    - cv: Heat capacity at 298.15K (cal/mol/K)
    """
    
    # Property names and their indices in QM9
    PROPERTY_NAMES = [
        'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve',
        'u0', 'u298', 'h298', 'g298', 'cv'
    ]
    
    # Units for each property
    PROPERTY_UNITS = {
        'mu': 'Debye',
        'alpha': 'Bohr^3', 
        'homo': 'Hartree',
        'lumo': 'Hartree',
        'gap': 'Hartree',
        'r2': 'Bohr^2',
        'zpve': 'Hartree',
        'u0': 'Hartree',
        'u298': 'Hartree', 
        'h298': 'Hartree',
        'g298': 'Hartree',
        'cv': 'cal/mol/K'
    }
    
    def __init__(
        self,
        data_dir: str,
        subset: str = 'train',
        target_properties: Optional[List[str]] = None,
        transform: Optional[callable] = None,
        use_features: List[str] = ['descriptors', 'fingerprints'],
        cache_features: bool = True
    ):
        """
        Initialize QM9Dataset.
        
        Args:
            data_dir: Path to directory containing QM9 data files
            subset: Dataset subset ('train', 'val', 'test')
            target_properties: List of target properties to predict
            transform: Optional transform to apply to molecular representations
            use_features: List of feature types to use ('descriptors', 'fingerprints', 'coordinates')
            cache_features: Whether to cache computed features
        """
        self.data_dir = data_dir
        self.subset = subset
        self.target_properties = target_properties or self.PROPERTY_NAMES
        self.transform = transform
        self.use_features = use_features
        self.cache_features = cache_features
        
        # Validate target properties
        invalid_props = set(self.target_properties) - set(self.PROPERTY_NAMES)
        if invalid_props:
            raise ValueError(f"Invalid target properties: {invalid_props}")
        
        # Load data
        self._load_data()
        
        # Cache for computed features
        self._feature_cache = {} if cache_features else None
        
    def _load_data(self):
        """Load QM9 data from files."""
        try:
            # Load the data - this would typically load from SDF files or preprocessed format
            self.molecules, self.properties = load_qm9_data(
                self.data_dir, subset=self.subset
            )
        except FileNotFoundError:
            # For demonstration, create dummy data structure
            print(f"Warning: QM9 data not found in {self.data_dir}. Using dummy data.")
            self._create_dummy_data()
    
    def _create_dummy_data(self):
        """Create dummy data for demonstration purposes."""
        n_samples = 1000 if self.subset == 'train' else 200
        
        # Dummy SMILES strings (simple molecules)
        dummy_smiles = [
            'C', 'CC', 'CCC', 'CCCC', 'CCO', 'CCN', 'C=C', 'C#C',
            'c1ccccc1', 'CCc1ccccc1', 'CCO', 'CCC(=O)O', 'CCN(C)C'
        ]
        
        self.molecules = {
            'smiles': [dummy_smiles[i % len(dummy_smiles)] for i in range(n_samples)],
            'mol_id': list(range(n_samples))
        }
        
        # Dummy properties with realistic ranges
        property_ranges = {
            'mu': (0.0, 10.0),
            'alpha': (10.0, 200.0),
            'homo': (-0.5, -0.1),
            'lumo': (-0.1, 0.3),
            'gap': (0.05, 0.5),
            'r2': (30.0, 2000.0),
            'zpve': (0.0, 0.5),
            'u0': (-1000.0, -50.0),
            'u298': (-1000.0, -50.0),
            'h298': (-1000.0, -50.0),
            'g298': (-1000.0, -50.0),
            'cv': (5.0, 50.0)
        }
        
        np.random.seed(42 if self.subset == 'train' else 123)
        self.properties = {}
        for prop in self.PROPERTY_NAMES:
            min_val, max_val = property_ranges[prop]
            self.properties[prop] = np.random.uniform(
                min_val, max_val, size=n_samples
            )
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.molecules['smiles'])
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing features and target properties
        """
        # Get molecule data
        smiles = self.molecules['smiles'][idx]
        mol_id = self.molecules['mol_id'][idx]
        
        # Compute or retrieve cached features
        if self._feature_cache is not None and idx in self._feature_cache:
            features = self._feature_cache[idx]
        else:
            features = self._compute_features(smiles, mol_id)
            if self._feature_cache is not None:
                self._feature_cache[idx] = features
        
        # Get target properties
        targets = torch.tensor([
            self.properties[prop][idx] for prop in self.target_properties
        ], dtype=torch.float32)
        
        # Apply transform if provided
        if self.transform:
            features = self.transform(features)
        
        return {
            'features': features,
            'targets': targets,
            'smiles': smiles,
            'mol_id': mol_id,
            'property_names': self.target_properties
        }
    
    def _compute_features(self, smiles: str, mol_id: int) -> torch.Tensor:
        """
        Compute molecular features from SMILES.
        
        Args:
            smiles: SMILES string
            mol_id: Molecule ID
            
        Returns:
            Feature tensor
        """
        from ..features import MolecularFeaturizer
        
        featurizer = MolecularFeaturizer(feature_types=self.use_features)
        features = featurizer.featurize(smiles)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def get_property_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for each target property.
        
        Returns:
            Dictionary with mean, std, min, max for each property
        """
        stats = {}
        for prop in self.target_properties:
            values = self.properties[prop]
            stats[prop] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        return stats
    
    def normalize_properties(self, method: str = 'standardize'):
        """
        Normalize target properties.
        
        Args:
            method: Normalization method ('standardize' or 'minmax')
        """
        if method == 'standardize':
            for prop in self.target_properties:
                values = self.properties[prop]
                mean = np.mean(values)
                std = np.std(values)
                self.properties[prop] = (values - mean) / (std + 1e-8)
        elif method == 'minmax':
            for prop in self.target_properties:
                values = self.properties[prop]
                min_val = np.min(values)
                max_val = np.max(values)
                self.properties[prop] = (values - min_val) / (max_val - min_val + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")