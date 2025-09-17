import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom
import torch



class QM9DataExtractor:
    """QM9数据集提取器"""

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.property_columns = ['A', 'B', 'C', 'miu', 'alpha', 'homo', 'lumo',
                                'gap', 'R2', 'zpve', 'Uo', 'U', 'H', 'G', 'Cv']

    def get_molecule_info(self, identifier):
        """获取完整分子信息"""
        # 解析标识符为索引
        if isinstance(identifier, str):
            mask = self.df['smiles_gdb'] == identifier
            if not mask.any():
                raise ValueError(f"SMILES '{identifier}' not found")
            idx = mask.idxmax()
        elif isinstance(identifier, int):
            if identifier < 0 or identifier >= len(self.df):
                raise IndexError(f"Index {identifier} out of range")
            idx = identifier
        else:
            raise TypeError("Identifier must be int or str")

        row = self.df.iloc[idx]
        smiles = row['smiles_gdb']

        # 解析坐标信息（数据已在提取时处理过科学计数法）
        atom_coords_str = row['atom_coords']
        lines = atom_coords_str.strip().split('\n')
        atoms = []
        coordinates = []
        charges = []

        for line in lines:
            parts = line.split('\t')
            atoms.append(parts[0])
            coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])
            charges.append(float(parts[4]))

        # 解析属性信息（数据已在提取时处理过科学计数法）
        properties = {}
        property_vector = []
        for prop in self.property_columns:
            value = float(row[prop])
            properties[prop] = value
            property_vector.append(value)

        return {
            'coordinates_data': {
                'smiles_gdb': smiles,
                'atoms': atoms,
                'coordinates': np.array(coordinates),
                'charges': np.array(charges),
                'num_atoms': len(atoms)
            },
            'properties_data': {
                'smiles_gdb': smiles,
                'properties': properties,
                'property_vector': np.array(property_vector)
            }
        }

    def get_smiles_list(self):
        """获取所有SMILES字符串列表"""
        return self.df['smiles_gdb'].tolist()

    def get_data_by_smiles(self, target_smiles):
        """通过SMILES字符串获取分子数据"""
        try:
            return self.get_molecule_info(target_smiles)
        except ValueError:
            return None


class MoleculeToGraph:
    """分子到图结构转换器"""
    
    def __init__(self):
        self.atom_types = ['H', 'C', 'N', 'O', 'F']
        self.hybridization_types = [
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED,
        ]
        self.bond_types = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]
    
    def convert_to_graph(self, molecule_data, smiles=None):
        """将分子数据转换为图结构"""
        coords_data = molecule_data['coordinates_data']
        atoms = coords_data['atoms']
        coordinates = coords_data['coordinates']
        charges = coords_data['charges']
        num_atoms = len(atoms)

        # 尝试创建RDKit分子对象
        mol = None
        if smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    mol = Chem.AddHs(mol)
                    rdDistGeom.EmbedMolecule(mol, randomSeed=42)
            except:
                mol = None

        # 构建节点特征
        node_features = []
        for i, (atom_symbol, coord, charge) in enumerate(zip(atoms, coordinates, charges)):
            rdkit_atom = mol.GetAtomWithIdx(i) if mol else None
            
            # 原子类型 one-hot编码
            atom_onehot = [0] * len(self.atom_types)
            if atom_symbol in self.atom_types:
                atom_onehot[self.atom_types.index(atom_symbol)] = 1

            # 杂化类型 one-hot编码
            hybrid_onehot = [0] * len(self.hybridization_types)
            if rdkit_atom and rdkit_atom.GetHybridization() in self.hybridization_types:
                hybrid_onehot[self.hybridization_types.index(rdkit_atom.GetHybridization())] = 1

            # 组合特征：原子类型 + 坐标 + 电荷 + 度数 + 杂化类型
            node_feat = (atom_onehot + coord.tolist() + [charge] +
                        [rdkit_atom.GetDegree() if rdkit_atom else 0] + hybrid_onehot)
            node_features.append(node_feat)
        
        # 构建边特征
        edge_indices = []
        edge_features = []
        
        if mol:
            # 使用RDKit化学键信息
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                distance = np.linalg.norm(coordinates[i] - coordinates[j])

                # 键类型 one-hot编码
                bond_onehot = [0] * len(self.bond_types)
                if bond.GetBondType() in self.bond_types:
                    bond_onehot[self.bond_types.index(bond.GetBondType())] = 1

                edge_feat = ([distance] + bond_onehot +
                           [1 if bond.GetIsConjugated() else 0] +
                           [1 if bond.IsInRing() else 0])

                # 添加双向边
                edge_indices.extend([[i, j], [j, i]])
                edge_features.extend([edge_feat, edge_feat])
        else:
            # 基于距离阈值创建边
            distance_threshold = 2.0
            for i in range(num_atoms):
                for j in range(i + 1, num_atoms):
                    distance = np.linalg.norm(coordinates[i] - coordinates[j])
                    if distance <= distance_threshold:
                        edge_feat = [distance] + [0] * (len(self.bond_types) + 2)
                        edge_indices.extend([[i, j], [j, i]])
                        edge_features.extend([edge_feat, edge_feat])
        
        # 处理空边情况
        if edge_indices:
            edge_index = np.array(edge_indices).T
            edge_features = np.array(edge_features)
        else:
            edge_index = np.zeros((2, 0), dtype=int)
            edge_features = np.zeros((0, 1 + len(self.bond_types) + 2))
        
        return {
            'x': torch.tensor(node_features, dtype=torch.float),
            'edge_index': torch.tensor(edge_index, dtype=torch.long),
            'edge_attr': torch.tensor(edge_features, dtype=torch.float),
            'num_nodes': len(node_features),
            'num_edges': len(edge_features)
        }
