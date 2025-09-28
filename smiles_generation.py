from rdkit import Chem
from rdkit.Chem import rdmolops
from itertools import combinations_with_replacement
from collections import defaultdict


def _build_r4n_smiles(substituents):
    """构建季铵离子SMILES字符串"""
    if len(substituents) != 4:
        return None

    # 将第一个替基作为主链，其他作为分支
    main_chain = substituents[0]
    branches = substituents[1:]

    # 构造 [N+](R1)(R2)(R3)R4 格式
    branch_part = ''.join(f'({r})' for r in branches)
    return f'[N+]{branch_part}{main_chain}'


def _validate_molecule(smiles):
    """验证分子结构的有效性"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return False, None

        # 检查分子电荷是否为+1
        if rdmolops.GetFormalCharge(mol) != 1:
            return False, None

        # 检查是否存在四价氮原子
        for atom in mol.GetAtoms():
            if (atom.GetSymbol() == 'N' and
                    atom.GetFormalCharge() == 1 and
                    atom.GetDegree() == 4):
                return True, mol

        return False, None

    except Exception:
        return False, None


def _get_canonical_info(mol):
    """获取标准化信息"""
    canonical_smiles = Chem.MolToSmiles(mol)
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
    return canonical_smiles, carbon_count


class R4NGenerator:
    """季铵离子(R4N+)化合物生成器"""

    def __init__(self, max_carbons):
        self.max_carbons = max_carbons
        self.alkyl_groups = self._generate_alkyl_groups()

    def _generate_alkyl_groups(self):
        """系统性生成烷基基团"""
        alkyls = []

        # 直链烷基 C1-C7
        for n in range(1, min(self.max_carbons + 1, 8)):
            alkyls.append(("C" * n, n))

        # 常见分支烷基（仅当碳数足够时添加）
        branched = [
            ("C(C)C", 3),  # 异丙基
            ("CC(C)C", 4),  # 异丁基
            ("C(C)(C)C", 4),  # 叔丁基
            ("CC(C)CC", 5),  # 异戊基
            ("CCC(C)C", 5),  # 仲戊基
            ("C(C)CCC", 5),  # 新戊基
        ]

        for smiles, carbons in branched:
            if carbons <= self.max_carbons:
                alkyls.append((smiles, carbons))

        return alkyls

    def generate_compounds(self):
        """生成所有可能的季铵离子化合物"""
        unique_compounds = set()
        carbon_distribution = defaultdict(int)

        print(f"Use {len(self.alkyl_groups)} alkyl groups to build R4N+ cations...")

        # 遍历所有4个烷基的组合（允许重复）
        for combo in combinations_with_replacement(self.alkyl_groups, 4):
            total_carbons = sum(alkyl[1] for alkyl in combo)

            # 跳过碳原子数超限的组合
            if total_carbons > self.max_carbons:
                continue

            # 提取烷基SMILES
            substituents = [alkyl[0] for alkyl in combo]

            # 构建季铵离子
            smiles = _build_r4n_smiles(substituents)
            if not smiles:
                continue

            # 验证分子结构
            is_valid, mol = _validate_molecule(smiles)
            if not is_valid:
                continue

            # 获取标准化信息
            canonical_smiles, carbon_count = _get_canonical_info(mol)

            # 添加到结果集（自动去重）
            if canonical_smiles not in {compound[1] for compound in unique_compounds}:
                unique_compounds.add((carbon_count, canonical_smiles))
                carbon_distribution[carbon_count] += 1

        return sorted(unique_compounds), dict(carbon_distribution)

    def save_results(self, compounds, filename=None):
        """保存结果到CSV文件"""
        if filename is None:
            filename = f"dataset_r4n_c{self.max_carbons}.csv"

        # 计算索引的宽度：与总条数的位数一致，例如 1-9 => 1 位，10-99 => 2 位
        total = len(compounds)
        index_width = max(1, len(str(total)))

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("Index,Num_c,SMILES\n")
            for i, (carbon_count, smiles) in enumerate(compounds, 1):
                # 使用前导零补齐，例如 0001、0002
                index_str = f"{i:0{index_width}d}"
                f.write(f"{index_str},{carbon_count},{smiles}\n")

        print(f"Result has been saved to {filename}")
        return filename


def print_statistics(compounds, carbon_distribution):
    """打印统计信息"""
    total_compounds = len(compounds)
    print(f"\nGenerating reports:")
    print(f"Total R4N+ Cation: {total_compounds}")
    print(f"Distribution of the num of carbon atoms:")

    for carbons in sorted(carbon_distribution.keys()):
        count = carbon_distribution[carbons]
        percentage = (count / total_compounds * 100) if total_compounds > 0 else 0
        print(f"  {carbons}C: {count} ({percentage:.1f}%)")


def main():
    print("R4N+ Cation Generator")
    print("LET'S GO, Damn!")
    print("=" * 50)

    # 获取用户输入
    while True:
        try:
            max_carbons = int(input("Plz input the max num of carbon atoms (Suggestion: 4-20): "))
            if max_carbons < 4:
                print("Error: R4N+ needs at least 4 carbons.")
                continue
            if max_carbons > 30:
                confirm = input("Warning: Too many max_carbons might take long time, continue？(y/n): ")
                if confirm.lower() != 'y':
                    continue
            break
        except ValueError:
            print("Error: plz input a valid integer.")

    # 生成化合物
    generator = R4NGenerator(max_carbons)
    compounds, carbon_distribution = generator.generate_compounds()

    # 显示统计信息
    print_statistics(compounds, carbon_distribution)

    # 保存结果
    generator.save_results(compounds, filename=f'data/r4n_smiles_c{max_carbons}.csv')

    print(f"\n{len(compounds)} R4N+ Cation Generated.")


if __name__ == "__main__":
    main()
    # 释放资源
    import gc
    gc.collect()