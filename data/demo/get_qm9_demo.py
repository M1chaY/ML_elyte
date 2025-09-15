import pandas as pd
from tqdm import tqdm
import os

# 属性列名
prop_names = ['mol_id', 'A', 'B', 'C', 'miu', 'alpha', 'homo', 'lumo', 'gap', 'R2', 'zpve', 'Uo', 'U', 'H', 'G', 'Cv']

# 生成文件路径列表（1-2000）
file_paths = [f'../raw/qm9_xyz/dsgdb9nsd_{i:06d}.xyz' for i in range(1, 501)]

# 存储数据
data_list = []

# 处理文件
for filepath in tqdm(file_paths, desc="提取测试数据"):
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        continue

    try:
        with open(filepath, 'r') as f:
            lines = f.read().strip().split('\n')

        # 解析属性
        props = lines[1].split()

        # 解析SMILES和InChI
        smiles_line = lines[-2].split('\t')
        inchi_line = lines[-1].split('\t')

        # 构建数据
        mol_data = {
            'file_name': filepath.split('/')[-1],
            'mol_len': int(lines[0]),
            'atom_coords': '\n'.join(lines[2:-3]),
            'vibrationalfrequence': lines[-3],
            'smiles_basic': smiles_line[0] if smiles_line else "",
            'smiles_stereo': smiles_line[1] if len(smiles_line) > 1 else "",
            'inchi_basic': inchi_line[0] if inchi_line else "",
            'inchi_stereo': inchi_line[1] if len(inchi_line) > 1 else ""
        }

        # 添加16个属性
        for i, name in enumerate(prop_names):
            mol_data[name] = props[0].split()[-1] if name == 'mol_id' else props[i + 1]

        data_list.append(mol_data)

    except Exception as e:
        print(f"处理文件 {filepath} 时出错: {e}")

# 保存结果
df = pd.DataFrame(data_list)
df = df.drop(columns=['mol_id'])

# 把InChI列中的 InChI= 去掉
df['inchi_basic'] = df['inchi_basic'].str.replace('InChI=', '', regex=False)
df['inchi_stereo'] = df['inchi_stereo'].str.replace('InChI=', '', regex=False)

df.to_csv('qm9_demo_0.5k.csv', index=False)

print(f"提取完成: {len(df)} 个分子")
print(f"数据结构: {df.shape}")
print("\n前5行预览:")
print(df[['file_name', 'mol_len', 'smiles_basic']].head())