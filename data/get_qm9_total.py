import pandas as pd
import glob
from tqdm import tqdm

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

        # 解析分子属性
        props = lines[1].split()

        # 解析SMILES（倒数第二行，两个版本）
        smiles_line = lines[-2].split('\t')
        smiles_basic = smiles_line[0] if len(smiles_line) > 0 else ""
        smiles_stereo = smiles_line[1] if len(smiles_line) > 1 else ""

        # 解析InChI（最后一行，两个版本）
        inchi_line = lines[-1].split('\t')
        inchi_basic = inchi_line[0] if len(inchi_line) > 0 else ""
        inchi_stereo = inchi_line[1] if len(inchi_line) > 1 else ""

        # 构建数据字典
        mol_data = {
            'file_name': filepath.split('/')[-1],  # 只保留文件名
            'mol_len': int(lines[0]),
            'atom_coords': '\n'.join(lines[2:-3]),
            'vibrationalfrequence': lines[-3],
            'smiles_basic': smiles_basic,
            'smiles_stereo': smiles_stereo,
            'inchi_basic': inchi_basic,
            'inchi_stereo': inchi_stereo
        }

        # 添加16个属性
        for i, name in enumerate(prop_names):
            mol_data[name] = props[i + 1] if name != 'mol_id' else props[0].split()[-1]

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
