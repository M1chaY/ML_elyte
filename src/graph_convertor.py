from utils.data import QM9DataExtractor
from utils.data import MoleculeToGraph
import torch
from tqdm import tqdm

# 使用定义的QM9DataExtractor类用于加载提取数据
extractor = QM9DataExtractor('../data/demo/qm9_demo.csv')

# 读取所有 smiles_gdb 数据
smiles_list = extractor.get_smiles_list()
print(f"数据集中分子数量: {len(smiles_list)}")

# 初始化分子图转换器
mol_to_graph = MoleculeToGraph()

# 存储所有图数据
graph_data_list = []
properties_list = []
failed_molecules = []

print("开始转换分子为图结构...")
for i, smiles in enumerate(tqdm(smiles_list, desc="转换进度")):
    try:
        # 获取分子完整信息
        molecule_info = extractor.get_molecule_info(i)

        # 转换为图结构
        graph_data = mol_to_graph.convert_to_graph(
            molecule_info,
            smiles=smiles
        )

        # 添加属性信息
        graph_data['y'] = torch.tensor(
            molecule_info['properties_data']['property_vector'],
            dtype=torch.float
        )
        graph_data['smiles'] = smiles

        graph_data_list.append(graph_data)
        properties_list.append(molecule_info['properties_data']['properties'])

    except Exception as e:
        failed_molecules.append((i, smiles, str(e)))
        print(f"分子 {i} ({smiles}) 转换失败: {e}")

print(f"\n转换完成!")
print(f"成功转换: {len(graph_data_list)} 个分子")
print(f"转换失败: {len(failed_molecules)} 个分子")