import pandas as pd
import numpy as np
import re


def safe_float_convert(value_str):
    """安全转换字符串为浮点数，处理科学计数法格式

    Args:
        value_str: 字符串形式的数值

    Returns:
        float: 转换后的浮点数
    """
    if isinstance(value_str, (int, float)):
        return float(value_str)

    value_str = str(value_str).strip()

    # 处理标准科学计数法 (e.g., 1.23e-4)
    try:
        return float(value_str)
    except ValueError:
        pass

    # 处理Mathematica风格的科学计数法 (e.g., 1.23*^-4)
    mathematica_pattern = r'([+-]?\d*\.?\d+)\*\^([+-]?\d+)'
    match = re.match(mathematica_pattern, value_str)
    if match:
        mantissa = float(match.group(1))
        exponent = int(match.group(2))
        return mantissa * (10 ** exponent)

    # 处理其他可能的格式
    # 移除空格和特殊字符
    cleaned = re.sub(r'[^\d.eE+-]', '', value_str)
    try:
        return float(cleaned)
    except ValueError:
        # 如果所有方法都失败，返回0.0并打印警告
        print(f"警告: 无法转换数值 '{value_str}'，使用默认值 0.0")
        return 0.0


class QM9DataExtractor:
    """QM9数据集提取器 - 用于提取分子坐标和属性信息"""

    def __init__(self, csv_file):
        """初始化数据提取器

        Args:
            csv_file: QM9数据集CSV文件路径
        """
        self.df = pd.read_csv(csv_file)
        # 分子属性列名列表
        self.property_columns = ['A', 'B', 'C', 'miu', 'alpha', 'homo', 'lumo', 'gap', 'R2', 'zpve', 'Uo', 'U', 'H',
                                 'G', 'Cv']

    def _get_idx_from_smiles(self, smiles):
        """通过SMILES字符串获取DataFrame索引"""
        mask = self.df['smiles_stereo'] == smiles
        if not mask.any():
            raise ValueError(f"SMILES '{smiles}' not found in dataset")
        return mask.idxmax()

    def _resolve_identifier(self, identifier):
        """解析标识符为DataFrame索引 - 支持整数索引和SMILES字符串"""
        if isinstance(identifier, str):
            return self._get_idx_from_smiles(identifier)
        elif isinstance(identifier, int):
            if identifier < 0 or identifier >= len(self.df):
                raise IndexError(f"Index {identifier} out of range for dataset of size {len(self.df)}")
            return identifier
        else:
            raise TypeError("Identifier must be either int (index) or str (SMILES)")

    def _parse_coords(self, idx):
        """解析单个分子的原子坐标信息"""
        atom_coords_str = self.df.iloc[idx]['atom_coords']
        smiles = self.df.iloc[idx]['smiles_stereo']

        lines = atom_coords_str.strip().split('\n')
        atoms = []
        coordinates = []
        charges = []

        # 解析每行原子信息: 原子类型 x y z 电荷
        for line in lines:
            parts = line.split('\t')
            atoms.append(parts[0])

            # 使用安全转换函数处理坐标
            x = safe_float_convert(parts[1])
            y = safe_float_convert(parts[2])
            z = safe_float_convert(parts[3])
            charge = safe_float_convert(parts[4])

            coordinates.append([x, y, z])
            charges.append(charge)

        return {
            'smiles': smiles,
            'atoms': atoms,
            'coordinates': np.array(coordinates),
            'charges': np.array(charges),
            'num_atoms': len(atoms)
        }

    def _parse_properties(self, idx):
        """解析单个分子的属性信息"""
        row = self.df.iloc[idx]
        smiles = row['smiles_stereo']

        # 提取所有属性值，使用安全转换
        properties = {}
        for prop in self.property_columns:
            properties[prop] = safe_float_convert(row[prop])

        return {
            'smiles': smiles,
            'properties': properties,
            'property_vector': np.array([safe_float_convert(row[prop]) for prop in self.property_columns])
        }

    def get_molecule_info(self, identifier):
        """获取完整分子信息(坐标+属性)

        Args:
            identifier: int (索引) 或 str (SMILES字符串)

        Returns:
            dict: 包含坐标和属性的完整分子信息
        """
        idx = self._resolve_identifier(identifier)

        # 获取坐标和属性信息
        coords_data = self._parse_coords(idx)
        properties_data = self._parse_properties(idx)

        return {
            'coordinates_data': coords_data,
            'properties_data': properties_data
        }

    def get_smiles_list(self):
        """获取所有SMILES字符串列表"""
        return self.df['smiles_stereo'].tolist()

    def get_data_by_smiles(self, target_smiles):
        """通过SMILES字符串获取分子数据"""
        try:
            return self.get_molecule_info(target_smiles)
        except ValueError:
            return None
