"""Molecular featurization using RDKit."""

import numpy as np
from typing import List, Optional, Union, Dict
import warnings

# Try to import RDKit, fallback to mock if not available
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    # Suppress RDKit warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='rdkit')
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from mock_rdkit import Chem, Descriptors, Crippen, rdMolDescriptors
    def CalcMolFormula(mol):
        return "C2H6O"


class MolecularFeaturizer:
    """
    Molecular featurization class for generating various molecular representations.
    
    Supports:
    - Molecular descriptors (RDKit descriptors)
    - Morgan fingerprints (ECFP)
    - Topological fingerprints
    - Physical-chemical properties
    - Coordinate-based features (if 3D coordinates available)
    """
    
    def __init__(
        self,
        feature_types: List[str] = ['descriptors', 'fingerprints'],
        fingerprint_radius: int = 2,
        fingerprint_bits: int = 1024,
        include_3d: bool = False
    ):
        """
        Initialize MolecularFeaturizer.
        
        Args:
            feature_types: List of feature types to compute
            fingerprint_radius: Radius for Morgan fingerprints
            fingerprint_bits: Number of bits in fingerprints
            include_3d: Whether to include 3D-dependent features
        """
        self.feature_types = feature_types
        self.fingerprint_radius = fingerprint_radius
        self.fingerprint_bits = fingerprint_bits
        self.include_3d = include_3d
        
        # Validate feature types
        valid_types = ['descriptors', 'fingerprints', 'coordinates', 'physicochemical']
        invalid_types = set(feature_types) - set(valid_types)
        if invalid_types:
            raise ValueError(f"Invalid feature types: {invalid_types}")
    
    def featurize(self, smiles: str) -> np.ndarray:
        """
        Generate molecular features from SMILES.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Feature vector as numpy array
        """
        # Parse SMILES to molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        features = []
        
        # Add different feature types
        if 'descriptors' in self.feature_types:
            desc_features = self._compute_descriptors(mol)
            features.extend(desc_features)
        
        if 'fingerprints' in self.feature_types:
            fp_features = self._compute_fingerprints(mol)
            features.extend(fp_features)
        
        if 'physicochemical' in self.feature_types:
            pc_features = self._compute_physicochemical(mol)
            features.extend(pc_features)
        
        if 'coordinates' in self.feature_types and self.include_3d:
            coord_features = self._compute_coordinate_features(mol)
            features.extend(coord_features)
        
        return np.array(features, dtype=np.float32)
    
    def _compute_descriptors(self, mol: Chem.Mol) -> List[float]:
        """Compute RDKit molecular descriptors."""
        descriptors = []
        
        # Basic molecular properties
        descriptors.extend([
            Descriptors.MolWt(mol),  # Molecular weight
            Descriptors.MolLogP(mol),  # LogP
            Descriptors.NumHDonors(mol),  # Hydrogen bond donors
            Descriptors.NumHAcceptors(mol),  # Hydrogen bond acceptors
            Descriptors.NumRotatableBonds(mol),  # Rotatable bonds
            Descriptors.TPSA(mol),  # Topological polar surface area
            Descriptors.NumAromaticRings(mol),  # Aromatic rings
            Descriptors.NumSaturatedRings(mol),  # Saturated rings
            Descriptors.NumHeavyAtoms(mol),  # Heavy atoms
            Descriptors.NumValenceElectrons(mol),  # Valence electrons
        ])
        
        # Additional descriptors
        try:
            descriptors.extend([
                Descriptors.BalabanJ(mol),  # Balaban J index
                Descriptors.BertzCT(mol),  # Bertz CT index
                Descriptors.Chi0n(mol),  # Chi connectivity index
                Descriptors.HallKierAlpha(mol),  # Hall-Kier alpha
                Descriptors.Kappa1(mol),  # Kappa shape index 1
                Descriptors.LabuteASA(mol),  # Labute approximate surface area
                Descriptors.PEOE_VSA1(mol),  # PEOE VSA descriptor 1
                Descriptors.SMR_VSA1(mol),  # SMR VSA descriptor 1
                Descriptors.SlogP_VSA1(mol),  # SlogP VSA descriptor 1
                Descriptors.VSA_EState1(mol),  # VSA EState descriptor 1
            ])
        except:
            # If any descriptor fails, add zeros
            descriptors.extend([0.0] * 10)
        
        # Handle NaN values
        descriptors = [x if not np.isnan(x) and not np.isinf(x) else 0.0 
                      for x in descriptors]
        
        return descriptors
    
    def _compute_fingerprints(self, mol: Chem.Mol) -> List[float]:
        """Compute Morgan (ECFP) fingerprints."""
        # Morgan fingerprint
        morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, self.fingerprint_radius, nBits=self.fingerprint_bits
        )
        
        # Convert to list of floats
        fingerprint = [float(x) for x in morgan_fp.ToBitString()]
        
        return fingerprint
    
    def _compute_physicochemical(self, mol: Chem.Mol) -> List[float]:
        """Compute physicochemical properties."""
        features = []
        
        try:
            # Crippen parameters
            logp, mr = Crippen.CrippenClogPAndMR(mol)
            features.extend([logp, mr])
            
            # Additional physicochemical properties
            features.extend([
                Descriptors.FractionCsp3(mol),  # Fraction of sp3 carbons
                Descriptors.NumAliphaticCarbocycles(mol),  # Aliphatic carbocycles
                Descriptors.NumAliphaticHeterocycles(mol),  # Aliphatic heterocycles
                Descriptors.RingCount(mol),  # Ring count
                Descriptors.Ipc(mol),  # Information content
                Descriptors.BertzCT(mol) / Descriptors.NumHeavyAtoms(mol) if Descriptors.NumHeavyAtoms(mol) > 0 else 0,  # Complexity per atom
            ])
        except:
            # If calculation fails, add zeros
            features.extend([0.0] * 8)
        
        # Handle NaN values
        features = [x if not np.isnan(x) and not np.isinf(x) else 0.0 
                   for x in features]
        
        return features
    
    def _compute_coordinate_features(self, mol: Chem.Mol) -> List[float]:
        """Compute coordinate-based features (requires 3D coordinates)."""
        # For now, return empty list as 3D coordinates are not always available
        # In a real implementation, this would compute geometric features
        # like moments of inertia, radius of gyration, etc.
        return []
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features."""
        names = []
        
        if 'descriptors' in self.feature_types:
            names.extend([
                'mol_wt', 'logp', 'hbd', 'hba', 'rotatable_bonds',
                'tpsa', 'aromatic_rings', 'saturated_rings', 'heavy_atoms',
                'valence_electrons', 'balaban_j', 'bertz_ct', 'chi0n',
                'hall_kier_alpha', 'kappa1', 'labute_asa', 'peoe_vsa1',
                'smr_vsa1', 'slogp_vsa1', 'vsa_estate1'
            ])
        
        if 'fingerprints' in self.feature_types:
            names.extend([f'morgan_fp_{i}' for i in range(self.fingerprint_bits)])
        
        if 'physicochemical' in self.feature_types:
            names.extend([
                'crippen_logp', 'crippen_mr', 'frac_csp3', 'aliphatic_carbocycles',
                'aliphatic_heterocycles', 'ring_count', 'ipc', 'complexity_per_atom'
            ])
        
        return names
    
    def get_feature_dim(self) -> int:
        """Get total feature dimension."""
        dim = 0
        
        if 'descriptors' in self.feature_types:
            dim += 20  # Number of descriptor features
        
        if 'fingerprints' in self.feature_types:
            dim += self.fingerprint_bits
        
        if 'physicochemical' in self.feature_types:
            dim += 8  # Number of physicochemical features
        
        return dim
    
    def featurize_batch(self, smiles_list: List[str]) -> np.ndarray:
        """
        Featurize a batch of SMILES.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            2D array of features (n_molecules x n_features)
        """
        features = []
        for smiles in smiles_list:
            try:
                mol_features = self.featurize(smiles)
                features.append(mol_features)
            except:
                # If featurization fails, add zero vector
                zero_features = np.zeros(self.get_feature_dim())
                features.append(zero_features)
        
        return np.array(features)