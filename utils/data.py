import pandas as pd
import numpy as np


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
            # 首先尝试在smiles_gdb列中查找
            mask = self.df['smiles_gdb'] == identifier
            if not mask.any():
                # 如果在smiles_gdb中没找到，尝试在smiles_b3lyp中查找
                mask = self.df['smiles_b3lyp'] == identifier
                if not mask.any():
                    raise ValueError(f"SMILES '{identifier}' not found in either smiles_gdb or smiles_b3lyp")
            idx = mask.idxmax()
        elif isinstance(identifier, int):
            if identifier < 0 or identifier >= len(self.df):
                raise IndexError(f"Index {identifier} out of range")
            idx = identifier
        else:
            raise TypeError("Identifier must be int or str")

        row = self.df.iloc[idx]
        smiles_gdb = row['smiles_gdb']
        smiles_b3lyp = row['smiles_b3lyp']

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
                'smiles_gdb': smiles_gdb,
                'smiles_b3lyp': smiles_b3lyp,
                'atoms': atoms,
                'coordinates': np.array(coordinates),
                'charges': np.array(charges),
                'num_atoms': len(atoms)
            },
            'properties_data': {
                'smiles_gdb': smiles_gdb,
                'smiles_b3lyp': smiles_b3lyp,
                'properties': properties,
                'property_vector': np.array(property_vector)
            }
        }

    def get_smiles_list(self, source='b3lyp'):
        """获取所有SMILES字符串列表"""
        return self.df[f'smiles_{source}'].tolist()

    def get_data_by_smiles(self, target_smiles):
        """通过SMILES字符串获取分子数据"""
        try:
            return self.get_molecule_info(target_smiles)
        except ValueError:
            return None