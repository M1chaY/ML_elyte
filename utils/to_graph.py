import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdDistGeom


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

        # 尝试创建RDKit分子对象，但添加更严格的检查
        mol = None
        use_rdkit = False

        if smiles:
            try:
                # 先不添加氢原子，检查原子数量是否匹配
                mol_temp = Chem.MolFromSmiles(smiles)
                if mol_temp and mol_temp.GetNumAtoms() == num_atoms:
                    mol = Chem.AddHs(mol_temp)
                    # 再次检查添加氢后的原子数量
                    if mol.GetNumAtoms() == num_atoms:
                        rdDistGeom.EmbedMolecule(mol, randomSeed=42)
                        use_rdkit = True
                    else:
                        # 如果添加氢后数量不匹配，使用不添加氢的版本
                        mol = mol_temp
                        if mol.GetNumAtoms() == num_atoms:
                            use_rdkit = True
                        else:
                            mol = None
                            use_rdkit = False
                else:
                    mol = None
                    use_rdkit = False
            except Exception as e:
                mol = None
                use_rdkit = False
                print(f"RDKit处理SMILES失败: {smiles}, 错误: {str(e)}")

        # 构建节点特征
        node_features = []
        for i, (atom_symbol, coord, charge) in enumerate(zip(atoms, coordinates, charges)):
            rdkit_atom = None
            if use_rdkit and mol and i < mol.GetNumAtoms():
                try:
                    rdkit_atom = mol.GetAtomWithIdx(i)
                except:
                    rdkit_atom = None

            # 原子类型 one-hot编码
            atom_onehot = [0] * len(self.atom_types)
            if atom_symbol in self.atom_types:
                atom_onehot[self.atom_types.index(atom_symbol)] = 1

            # 杂化类型 one-hot编码
            hybrid_onehot = [0] * len(self.hybridization_types)
            if rdkit_atom:
                try:
                    if rdkit_atom.GetHybridization() in self.hybridization_types:
                        hybrid_onehot[self.hybridization_types.index(rdkit_atom.GetHybridization())] = 1
                except:
                    pass

            # 组合特征：原子类型 + 坐标 + 电荷 + 度数 + 杂化类型
            degree = rdkit_atom.GetDegree() if rdkit_atom else 0
            node_feat = (atom_onehot + coord.tolist() + [charge] + [degree] + hybrid_onehot)
            node_features.append(node_feat)

        # 构建边特征
        edge_indices = []
        edge_features = []

        if use_rdkit and mol:
            # 使用RDKit化学键信息
            for bond in mol.GetBonds():
                try:
                    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

                    # 确保索引在有效范围内
                    if i >= num_atoms or j >= num_atoms:
                        continue

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
                except Exception as e:
                    # 跳过有问题的键
                    continue
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
            'num_edges': len(edge_features),
            'used_rdkit': use_rdkit  # 添加标记表示是否使用了RDKit信息
        }