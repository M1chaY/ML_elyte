"""Molecular data utilities."""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import csv

# Try to import RDKit, fallback to mock if not available
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from mock_rdkit import Chem, Descriptors


def load_qm9_data(
    data_dir: str,
    subset: str = 'train',
    file_format: str = 'csv'
) -> Tuple[Dict, Dict]:
    """
    Load QM9 dataset from various formats.
    
    Args:
        data_dir: Directory containing QM9 data files
        subset: Data subset to load ('train', 'val', 'test', 'all')
        file_format: Data format ('csv', 'sdf', 'pkl')
        
    Returns:
        Tuple of (molecules, properties) dictionaries
    """
    if file_format == 'csv':
        return _load_qm9_csv(data_dir, subset)
    elif file_format == 'sdf':
        return _load_qm9_sdf(data_dir, subset)
    elif file_format == 'pkl':
        return _load_qm9_pickle(data_dir, subset)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def _load_qm9_csv(data_dir: str, subset: str) -> Tuple[Dict, Dict]:
    """Load QM9 data from CSV files."""
    # Expected CSV structure:
    # mol_id, smiles, mu, alpha, homo, lumo, gap, r2, zpve, u0, u298, h298, g298, cv
    
    if subset == 'all':
        csv_files = ['qm9_train.csv', 'qm9_val.csv', 'qm9_test.csv']
    else:
        csv_files = [f'qm9_{subset}.csv']
    
    molecules = {'smiles': [], 'mol_id': []}
    properties = {
        'mu': [], 'alpha': [], 'homo': [], 'lumo': [], 'gap': [],
        'r2': [], 'zpve': [], 'u0': [], 'u298': [], 'h298': [], 'g298': [], 'cv': []
    }
    
    for csv_file in csv_files:
        csv_path = os.path.join(data_dir, csv_file)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                molecules['smiles'].append(row['smiles'])
                molecules['mol_id'].append(int(row['mol_id']))
                
                for prop in properties.keys():
                    properties[prop].append(float(row[prop]))
    
    # Convert to numpy arrays
    for prop in properties.keys():
        properties[prop] = np.array(properties[prop])
    
    return molecules, properties


def _load_qm9_sdf(data_dir: str, subset: str) -> Tuple[Dict, Dict]:
    """Load QM9 data from SDF files."""
    # This would implement SDF file parsing
    # For now, raise not implemented
    raise NotImplementedError("SDF loading not yet implemented")


def _load_qm9_pickle(data_dir: str, subset: str) -> Tuple[Dict, Dict]:
    """Load QM9 data from pickle files."""
    import pickle
    
    pickle_path = os.path.join(data_dir, f'qm9_{subset}.pkl')
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['molecules'], data['properties']


def parse_smiles(smiles: str) -> Optional[Chem.Mol]:
    """
    Parse SMILES string to RDKit molecule object.
    
    Args:
        smiles: SMILES string
        
    Returns:
        RDKit Mol object or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Add hydrogens and compute 2D coordinates
            mol = Chem.AddHs(mol)
        return mol
    except:
        return None


def validate_smiles(smiles_list: List[str]) -> Dict[str, List]:
    """
    Validate a list of SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        Dictionary with valid and invalid SMILES
    """
    valid_smiles = []
    invalid_smiles = []
    
    for i, smiles in enumerate(smiles_list):
        mol = parse_smiles(smiles)
        if mol is not None:
            valid_smiles.append((i, smiles))
        else:
            invalid_smiles.append((i, smiles))
    
    return {
        'valid': valid_smiles,
        'invalid': invalid_smiles,
        'valid_count': len(valid_smiles),
        'invalid_count': len(invalid_smiles),
        'total_count': len(smiles_list)
    }


def standardize_smiles(smiles: str) -> str:
    """
    Standardize SMILES string using RDKit.
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        Standardized SMILES string
    """
    mol = parse_smiles(smiles)
    if mol is None:
        return smiles  # Return original if parsing fails
    
    # Standardize the molecule
    try:
        # Remove hydrogens, sanitize, and get canonical SMILES
        mol = Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
        standardized_smiles = Chem.MolToSmiles(mol, canonical=True)
        return standardized_smiles
    except:
        return smiles  # Return original if standardization fails


def compute_molecular_properties(smiles: str) -> Dict[str, float]:
    """
    Compute basic molecular properties from SMILES.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary of computed properties
    """
    mol = parse_smiles(smiles)
    if mol is None:
        return {}
    
    properties = {}
    try:
        properties['mol_wt'] = Descriptors.MolWt(mol)
        properties['logp'] = Descriptors.MolLogP(mol)
        properties['hbd'] = Descriptors.NumHDonors(mol)
        properties['hba'] = Descriptors.NumHAcceptors(mol)
        properties['rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
        properties['tpsa'] = Descriptors.TPSA(mol)
        properties['heavy_atoms'] = Descriptors.NumHeavyAtoms(mol)
    except:
        pass  # Return partial results if some calculations fail
    
    return properties


def save_predictions(
    predictions: np.ndarray,
    smiles: List[str],
    property_names: List[str],
    output_path: str,
    mol_ids: Optional[List[int]] = None,
    include_smiles: bool = True
):
    """
    Save predictions to CSV file.
    
    Args:
        predictions: Prediction array (n_samples, n_properties)
        smiles: List of SMILES strings
        property_names: Names of predicted properties
        output_path: Output CSV file path
        mol_ids: Optional molecule IDs
        include_smiles: Whether to include SMILES in output
    """
    # Prepare data
    data = {}
    
    if mol_ids is not None:
        data['mol_id'] = mol_ids
    else:
        data['mol_id'] = list(range(len(smiles)))
    
    if include_smiles:
        data['smiles'] = smiles
    
    # Add predictions
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    
    for i, prop_name in enumerate(property_names):
        if i < predictions.shape[1]:
            data[f'pred_{prop_name}'] = predictions[:, i]
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")


def create_qm9_splits(
    molecules: Dict,
    properties: Dict,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Dict[str, Tuple[Dict, Dict]]:
    """
    Create train/validation/test splits for QM9 data.
    
    Args:
        molecules: Dictionary with molecular data
        properties: Dictionary with property data
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with split data
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    np.random.seed(random_seed)
    n_total = len(molecules['smiles'])
    indices = np.random.permutation(n_total)
    
    # Calculate split sizes
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val
    
    # Create index splits
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    splits = {}
    
    for split_name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        # Create molecule subset
        split_molecules = {
            'smiles': [molecules['smiles'][i] for i in idx],
            'mol_id': [molecules['mol_id'][i] for i in idx]
        }
        
        # Create property subset
        split_properties = {}
        for prop_name, prop_values in properties.items():
            split_properties[prop_name] = prop_values[idx]
        
        splits[split_name] = (split_molecules, split_properties)
    
    print(f"Created splits - Train: {n_train}, Val: {n_val}, Test: {n_test}")
    return splits


def save_qm9_splits(
    splits: Dict[str, Tuple[Dict, Dict]],
    output_dir: str,
    file_format: str = 'csv'
):
    """
    Save QM9 data splits to files.
    
    Args:
        splits: Split data dictionary
        output_dir: Output directory
        file_format: Output format ('csv' or 'pkl')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, (molecules, properties) in splits.items():
        if file_format == 'csv':
            # Save as CSV
            data = {
                'mol_id': molecules['mol_id'],
                'smiles': molecules['smiles']
            }
            data.update(properties)
            
            df = pd.DataFrame(data)
            output_path = os.path.join(output_dir, f'qm9_{split_name}.csv')
            df.to_csv(output_path, index=False)
            
        elif file_format == 'pkl':
            # Save as pickle
            data = {'molecules': molecules, 'properties': properties}
            output_path = os.path.join(output_dir, f'qm9_{split_name}.pkl')
            
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
        
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        
        print(f"Saved {split_name} split to: {output_path}")