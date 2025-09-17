import pandas as pd
import glob
import re
from tqdm import tqdm
from typing import Dict, Any


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

    # Mathematica风格科学计数法 (1.*^-4, 1.23*^-4, -7.*^-6等)
    mathematica_match = re.match(r'([+-]?\d+\.?\d*)\*\^([+-]?\d+)', value_str)
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


# 存储所有数据
data_list = []

# 获取所有xyz文件
xyz_files = glob.glob('raw/qm9_xyz/*.xyz')
print(f"找到 {len(xyz_files)} 个xyz文件")

# 实际的属性名称（不包括gdb标识符和分子ID）
# 这15个属性对应props[2]到props[16]
actual_prop_names = ['A', 'B', 'C', 'miu', 'alpha', 'homo', 'lumo', 'gap',
                     'R2', 'zpve', 'Uo', 'U', 'H', 'G', 'Cv']

# 读取每个xyz文件
for filepath in tqdm(xyz_files, desc="处理文件"):
    try:
        with open(filepath, 'r') as f:
            lines = f.read().strip().split('\n')

        # 解析分子属性
        props = lines[1].split()

        # props[0] = 'gdb' (标识符)
        # props[1] = 分子ID (例如 '42')
        # props[2:17] = 15个属性值

        # 提取分子ID
        mol_id = int(props[1])

        # 解析SMILES（倒数第二行，两个版本）
        smiles_line = lines[-2].split('\t')
        smiles_basic = smiles_line[0] if len(smiles_line) > 0 else ""
        smiles_stereo = smiles_line[1] if len(smiles_line) > 1 else ""

        # 解析InChI（最后一行，两个版本）
        inchi_line = lines[-1].split('\t')
        inchi_basic = inchi_line[0] if len(inchi_line) > 0 else ""
        inchi_stereo = inchi_line[1] if len(inchi_line) > 1 else ""

        # 处理原子坐标数据
        coord_lines = lines[2:-3]  # 从第3行到倒数第4行
        processed_coords = []
        for coord_line in coord_lines:
            # 使用正则表达式分割，处理制表符和空格混合的情况
            parts = re.split(r'[\t\s]+', coord_line.strip())
            if len(parts) >= 5:
                atom_symbol = parts[0]
                x = _safe_float_convert(parts[1])
                y = _safe_float_convert(parts[2])
                z = _safe_float_convert(parts[3])
                charge = _safe_float_convert(parts[4])
                processed_coords.append(f"{atom_symbol}\t{x}\t{y}\t{z}\t{charge}")
            else:
                # 如果格式不对，保留原始行
                processed_coords.append(coord_line)

        # 构建数据字典 - 使用Any类型以兼容int、str和float
        mol_data: Dict[str, Any] = {
            # 'mol_id': mol_id,
            'smiles': smiles_stereo,
            # 'smiles_basic': smiles_basic,
            # 'mol_len': int(lines[0]),  # Atoms数量
            'atom_coords': '\n'.join(processed_coords),
            'vibrationalfrequence': lines[-3],  # 振动频率
            # 'inchi_basic': inchi_basic,
            # 'inchi_stereo': inchi_stereo,
            # 'file_name': filepath.split('/')[-1].replace('\\', '/')  # 处理Windows路径
        }

        # 添加15个属性值（全部为float类型）
        # props[2:17]对应actual_prop_names中的15个属性
        for i, name in enumerate(actual_prop_names):
            # i从0开始，对应props中从索引2开始
            prop_index = i + 2
            if prop_index < len(props):
                mol_data[name] = _safe_float_convert(props[prop_index])
            else:
                print(f"Warning: Missing property {name} in file {filepath}")
                mol_data[name] = 0.0

        data_list.append(mol_data)

    except Exception as e:
        print(f"处理文件 {filepath} 时出错: {e}")
        # 可选：打印更详细的调试信息
        import traceback

        traceback.print_exc()

# 转换为DataFrame并保存
df = pd.DataFrame(data_list)
print(f"\n处理完成: {len(df)} 个分子")
print(f"数据结构: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print("\n前5行数据预览:")
print(df[['smiles', 'A', 'B', 'C']].head())

# 保存为多种格式
df.to_csv('raw/qm9.csv', index=False)
print(f"\n数据已保存到 data/raw/qm9.csv")

# 如果需要保存为parquet格式（需要pyarrow或fastparquet）
# try:
#     df.to_parquet('qm9.parquet', compression='gzip')
#     print(f"数据已保存到 qm9.parquet")
# except Exception as e:
#     print(f"保存parquet格式失败（可能需要安装pyarrow）: {e}")

# 结束后释放内存
del df
del data_list
import gc
gc.collect()
print("内存已释放")