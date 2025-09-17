import pandas as pd
import glob
import re
from tqdm import tqdm


def _safe_float_convert(value_str):
    """安全转换字符串为浮点数，处理科学计数法格式

    Args:
        value_str: 字符串形式的数值

    Returns:
        float: 转换后的浮点数
    """
    if isinstance(value_str, (int, float)):
        return float(value_str)

    value_str = str(value_str).strip()

    # 标准科学计数法
    try:
        return float(value_str)
    except ValueError:
        pass

    # Mathematica风格科学计数法 (1.23*^-4)
    mathematica_match = re.match(r'([+-]?\d*\.?\d+)\*\^([+-]?\d+)', value_str)
    if mathematica_match:
        mantissa = float(mathematica_match.group(1))
        exponent = int(mathematica_match.group(2))
        return mantissa * (10 ** exponent)

    # 移除特殊字符后重试
    cleaned = re.sub(r'[^\d.eE+-]', '', value_str)
    try:
        return float(cleaned)
    except ValueError:
        print(f"Warning: Cannot convert '{value_str}', using default 0.0")
        return 0.0


# 属性列名
prop_names = ['mol_id', 'A', 'B', 'C', 'miu', 'alpha', 'homo', 'lumo', 'gap', 'R2', 'zpve', 'Uo', 'U', 'H', 'G', 'Cv']

# 存储所有数据
data_list = []

# 获取所有xyz文件
xyz_files = glob.glob('qm9_xyz/*.xyz')
print(f"找到 {len(xyz_files)} 个xyz文件")

# 读取每个xyz文件
for filepath in tqdm(xyz_files, desc="处理文件"):
    try:
        with open(filepath, 'r') as f:
            lines = f.read().strip().split('\n')

        # 解析分子属性 - 使用safe_float_convert处理科学计数法
        props = lines[1].split()

        # 解析SMILES（倒数第二行，两个版本）
        smiles_line = lines[-2].split('\t')
        smiles_basic = smiles_line[0] if len(smiles_line) > 0 else ""
        smiles_stereo = smiles_line[1] if len(smiles_line) > 1 else ""

        # 解析InChI（最后一行，两个版本）
        inchi_line = lines[-1].split('\t')
        inchi_basic = inchi_line[0] if len(inchi_line) > 0 else ""
        inchi_stereo = inchi_line[1] if len(inchi_line) > 1 else ""

        # 处理原子坐标数据 - 使用safe_float_convert处理科学计数法
        coord_lines = lines[2:-3]
        processed_coords = []
        for coord_line in coord_lines:
            parts = coord_line.split('\t')
            if len(parts) >= 5:
                atom_symbol = parts[0]
                x = _safe_float_convert(parts[1])
                y = _safe_float_convert(parts[2])
                z = _safe_float_convert(parts[3])
                charge = _safe_float_convert(parts[4])
                processed_coords.append(f"{atom_symbol}\t{x}\t{y}\t{z}\t{charge}")
            else:
                processed_coords.append(coord_line)

        # 构建数据字典
        mol_data = {
            'file_name': filepath.split('/')[-1],  # 只保留文件名
            'mol_len': int(lines[0]),
            'atom_coords': '\n'.join(processed_coords),
            'vibrationalfrequence': lines[-3],
            'smiles_basic': smiles_basic,
            'smiles_stereo': smiles_stereo,
            'inchi_basic': inchi_basic,
            'inchi_stereo': inchi_stereo
        }

        # 添加16个属性 - 使用safe_float_convert处理科学计数法
        for i, name in enumerate(prop_names):
            if name != 'mol_id':
                mol_data[name] = _safe_float_convert(props[i + 1])
            else:
                mol_data[name] = props[0].split()[-1]

        data_list.append(mol_data)

    except Exception as e:
        print(f"处理文件 {filepath} 时出错: {e}")

# 转换为DataFrame并保存
df = pd.DataFrame(data_list)
print(f"处理完成: {len(df)} 个分子")
print(f"数据结构: {df.shape}")
print("\n前5行数据:")
print(df[['file_name', 'mol_len', 'smiles_basic', 'smiles_stereo']].head())

# 保存为多种格式
df.to_csv('qm9.csv', index=False)
# df.to_parquet('qm9.parquet', compression='gzip')
