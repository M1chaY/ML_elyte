import pandas as pd
import numpy as np


class QM9DataExtractor:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.property_columns = ['A', 'B', 'C', 'miu', 'alpha', 'homo', 'lumo', 'gap', 'R2', 'zpve', 'Uo', 'U', 'H',
                                 'G', 'Cv']

    def extract_atom_coords(self, idx=None):
        """提取原子位置信息向量"""
        if idx is not None:
            return self._parse_single_coords(idx)

        coords_list = []
        for i in range(len(self.df)):
            coords_list.append(self._parse_single_coords(i))
        return coords_list

    def _parse_single_coords(self, idx):
        """解析单个分子的原子坐标"""
        atom_coords_str = self.df.iloc[idx]['atom_coords']
        smiles = self.df.iloc[idx]['smiles_basic']

        lines = atom_coords_str.strip().split('\n')
        atoms = []
        coordinates = []
        charges = []

        for line in lines:
            parts = line.split('\t')
            atoms.append(parts[0])
            coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])
            charges.append(float(parts[4]))

        return {
            'smiles': smiles,
            'atoms': atoms,
            'coordinates': np.array(coordinates),
            'charges': np.array(charges),
            'num_atoms': len(atoms)
        }

    def extract_properties(self, idx=None):
        """提取分子性质信息"""
        if idx is not None:
            return self._get_single_properties(idx)

        props_list = []
        for i in range(len(self.df)):
            props_list.append(self._get_single_properties(i))
        return props_list

    def _get_single_properties(self, idx):
        """获取单个分子的性质"""
        row = self.df.iloc[idx]
        smiles = row['smiles_basic']

        properties = {}
        for prop in self.property_columns:
            properties[prop] = row[prop]

        return {
            'smiles': smiles,
            'properties': properties,
            'property_vector': np.array([row[prop] for prop in self.property_columns])
        }

    def get_smiles_list(self):
        """获取所有SMILES字符串"""
        return self.df['smiles_basic'].tolist()

    def get_data_by_smiles(self, target_smiles):
        """根据SMILES字符串获取对应的坐标和性质"""
        mask = self.df['smiles_basic'] == target_smiles
        if not mask.any():
            return None

        idx = mask.idxmax()
        coords = self.extract_atom_coords(idx)
        props = self.extract_properties(idx)

        return {
            'coordinates_data': coords,
            'properties_data': props
        }


# 使用示例
if __name__ == "__main__":
    # 初始化提取器
    extractor = QM9DataExtractor('qm9_demo_0.5k.csv')

    # 提取第一个分子的信息
    coords_data = extractor.extract_atom_coords(0)
    props_data = extractor.extract_properties(0)

    print("第一个分子的坐标信息:")
    print(f"SMILES: {coords_data['smiles']}")
    print(f"原子类型: {coords_data['atoms']}")
    print(f"坐标矩阵形状: {coords_data['coordinates'].shape}")

    print("\n第一个分子的性质信息:")
    print(f"SMILES: {props_data['smiles']}")
    print(f"性质向量形状: {props_data['property_vector'].shape}")
    print(f"HOMO: {props_data['properties']['homo']}")
    print(f"LUMO: {props_data['properties']['lumo']}")
