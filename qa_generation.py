from rdkit import Chem
from rdkit.Chem import rdmolops
from itertools import combinations_with_replacement


def generate_alkyls(max_carbons):
    """生成烷基基团"""
    alkyls = []

    # 基本直链烷基
    for n in range(1, min(max_carbons + 1, 8)):
        alkyls.append(("C" * n, n))

    # 分支烷基
    if max_carbons >= 3:
        alkyls.append(("C(C)C", 3))  # 异丙基
    if max_carbons >= 4:
        alkyls.extend([("CC(C)C", 4), ("C(C)(C)C", 4)])  # 异丁基，叔丁基
    if max_carbons >= 5:
        alkyls.extend([("CC(C)CC", 5), ("CCC(C)C", 5), ("C(C)CCC", 5)])

    return alkyls


def build_quaternary_smiles(substituents):
    """正确构建季铵离子SMILES - 氮在中心连接四个基团"""
    # 对于季铵离子，格式应该是 [N+](R1)(R2)(R3)R4
    # 第一个基团不用括号，后面三个用括号
    if len(substituents) != 4:
        return None

    # 按长度排序，把最简单的基团放第一个（可选）
    sorted_subs = sorted(substituents)

    first = sorted_subs[0]
    others = sorted_subs[1:]

    # 构建SMILES
    other_parts = ''.join(f'({sub})' if len(sub) > 1 or '(' in sub else f'({sub})' for sub in others)
    smiles = f'[N+]{other_parts}{first}'

    return smiles


def test_quaternary_construction():
    """测试季铵离子构建"""
    print("=== 测试季铵离子构建 ===")

    test_cases = [
        (["C", "C", "C", "C"], "四甲基铵"),
        (["C", "C", "C", "CC"], "三甲基乙基铵"),
        (["C", "C", "CC", "CC"], "二甲基二乙基铵"),
        (["CC", "CC", "CC", "CC"], "四乙基铵"),
    ]

    for substituents, name in test_cases:
        smiles = build_quaternary_smiles(substituents)
        print(f"{name}: {substituents} -> {smiles}")

        if smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    charge = rdmolops.GetFormalCharge(mol)
                    carbons = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
                    canonical = Chem.MolToSmiles(mol)
                    print(f"  验证: {canonical} (C{carbons}, 电荷={charge})")

                    # 检查氮原子的连接数
                    for atom in mol.GetAtoms():
                        if atom.GetSymbol() == 'N':
                            degree = atom.GetDegree()
                            print(f"  氮原子连接数: {degree}")
                            break
                else:
                    print(f"  ✗ RDKit解析失败")
            except Exception as e:
                print(f"  ✗ 错误: {e}")
        print()


def generate_all_quaternary(max_carbons=12):
    """生成所有季铵阳离子"""
    alkyls = generate_alkyls(max_carbons)
    results = set()

    print(f"可用烷基: {[f'{a[0]}(C{a[1]})' for a in alkyls[:10]]}{'...' if len(alkyls) > 10 else ''}")

    for total_c in range(4, max_carbons + 1):
        print(f"生成 C{total_c}...")
        count = 0

        for combo in combinations_with_replacement(alkyls, 4):
            if sum(alkyl[1] for alkyl in combo) == total_c:
                substituents = [alkyl[0] for alkyl in combo]
                smiles = build_quaternary_smiles(substituents)

                if smiles:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol and rdmolops.GetFormalCharge(mol) == 1:
                            # 验证氮原子确实连接了4个基团
                            n_degree = 0
                            for atom in mol.GetAtoms():
                                if atom.GetSymbol() == 'N' and atom.GetFormalCharge() == 1:
                                    n_degree = atom.GetDegree()
                                    break

                            if n_degree == 4:  # 确保氮连接4个基团
                                canonical = Chem.MolToSmiles(mol)
                                carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')

                                if canonical not in [r[1] for r in results]:
                                    results.add((carbon_count, canonical))
                                    count += 1
                                    if count <= 3:  # 只显示前几个例子
                                        print(f"  {substituents} -> {canonical}")
                    except:
                        pass

        print(f"  C{total_c}: {count} 个")

    return sorted(list(results))


def main():
    # 首先测试构建逻辑
    test_quaternary_construction()

    # 生成季铵离子
    max_c = int(input("输入最大碳原子数: "))
    results = generate_all_quaternary(max_c)

    print(f"\n总共生成: {len(results)} 个季铵阳离子")

    # 统计
    from collections import Counter
    carbon_dist = Counter(r[0] for r in results)
    print("\n碳数分布:")
    for c in sorted(carbon_dist.keys()):
        print(f"  C{c}: {carbon_dist[c]} 个")

    # 显示全部结果
    print(f"\n所有季铵阳离子:")
    for i, (carbons, smiles) in enumerate(results, 1):
        print(f"{i:3d}. C{carbons}: {smiles}")

    # 保存
    # 用标准写入模式，确保兼容性
    with open("correct_quaternary_ammonium.csv", "w", encoding="utf-8") as f:
        f.write("序号,碳数,SMILES\n")
        for i, (carbons, smiles) in enumerate(results, 1):
            # 如需保存浮点数属性，建议用: f"{float_value:.8f}"
            f.write(f"{i},{carbons},{smiles}\n")

    print(f"\n保存到 correct_quaternary_ammonium.csv")


if __name__ == "__main__":
    main()