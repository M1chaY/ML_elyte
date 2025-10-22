import pandas as pd
import pubchempy as pcp
from typing import Optional
from rdkit import Chem

def add_halide_to_smiles(smiles, halide='Br'):
    """
    将季铵阳离子 SMILES 转化为卤化盐
    
    参数:
    smiles: str - 包含 R4N+ 的 SMILES 字符串
    halide: str - 'Br' 为溴化盐, 'Cl' 为氯化盐
    
    返回:
    str - 添加了卤素阴离子的 SMILES
    """
    if halide == 'Br':
        return smiles + '.[Br-]'
    elif halide == 'Cl':
        return smiles + '.[Cl-]'
    else:
        raise ValueError("halide must be 'Br' or 'Cl'")
    
def validate_single_salt(smiles: str, salt_type: str, verbose: bool = False) -> str:
    """
    验证单个盐的SMILES并获取CAS号
    
    参数:
        smiles: str - 盐的SMILES字符串
        salt_type: str - 'Bromide' 或 'Chloride'
        verbose: bool - 是否打印信息
    
    返回:
        str - CAS号,如果不存在返回空字符串
    """
    try:
        if verbose:
            print(f"  Querying {salt_type}...", end=' ', flush=True)
        
        results = pcp.get_compounds(smiles, 'smiles')
        
        if results:
            compound = results[0]
            synonyms = compound.synonyms if compound.synonyms else []
            
            # 筛选 CAS 号格式 (xxx-xx-x)
            cas_numbers = [s for s in synonyms if '-' in s and all(part.isdigit() for part in s.split('-'))]
            
            if cas_numbers:
                cas_str = ', '.join(cas_numbers)
                if verbose:
                    print(f"✓ CAS: {cas_str}", flush=True)
                return cas_str
        
        if verbose:
            print("✗ Not found", flush=True)
        return ''
        
    except Exception as e:
        if verbose:
            print(f"✗ Error: {str(e)}", flush=True)
        return ''


def validate_salts_and_add_cas_columns(df, filename):
    """
    为原始数据框添加溴盐CAS和氯盐CAS两列,并删除两个都不存在的行
    
    参数:
        df: DataFrame - 原始数据框
        filename: str - 文件名前缀
    
    返回:
        DataFrame - 添加了CAS列并过滤后的数据框
    """
    total = len(df)
    print(f"\n开始验证 {total} 个季铵阳离子的盐...\n")
    
    # 初始化CAS列
    bromide_cas_list = []
    chloride_cas_list = []
    
    # 遍历每一行
    for idx, row in df.iterrows():
        print(f"[{idx+1}/{total}] Processing Index {row['Index']}:", flush=True)
        
        smiles = row['SMILES']
        
        # 查询溴盐CAS
        bromide_smiles = add_halide_to_smiles(smiles, 'Br')
        bromide_cas = validate_single_salt(bromide_smiles, 'Bromide', verbose=True)
        bromide_cas_list.append(bromide_cas)
        
        # 查询氯盐CAS
        chloride_smiles = add_halide_to_smiles(smiles, 'Cl')
        chloride_cas = validate_single_salt(chloride_smiles, 'Chloride', verbose=True)
        chloride_cas_list.append(chloride_cas)
        
        print()  # 换行
    
    # 添加CAS列到原始数据框
    df['Bromide_CAS'] = bromide_cas_list
    df['Chloride_CAS'] = chloride_cas_list
    
    # 统计
    initial_count = len(df)
    both_exist = ((df['Bromide_CAS'] != '') & (df['Chloride_CAS'] != '')).sum()
    only_bromide = ((df['Bromide_CAS'] != '') & (df['Chloride_CAS'] == '')).sum()
    only_chloride = ((df['Bromide_CAS'] == '') & (df['Chloride_CAS'] != '')).sum()
    neither = ((df['Bromide_CAS'] == '') & (df['Chloride_CAS'] == '')).sum()
    
    print("\n" + "="*80)
    print("统计结果:")
    print(f"  两种盐都存在: {both_exist}")
    print(f"  仅溴盐存在:   {only_bromide}")
    print(f"  仅氯盐存在:   {only_chloride}")
    print(f"  两种都不存在: {neither}")
    print("="*80)
    
    # 删除两个CAS都不存在的行
    df_filtered = df[(df['Bromide_CAS'] != '') | (df['Chloride_CAS'] != '')].copy()
    
    # 保留原有的Index,不重新编号
    
    print(f"\n过滤前: {initial_count} 行")
    print(f"过滤后: {len(df_filtered)} 行 (删除了 {initial_count - len(df_filtered)} 行)\n")
    
    return df_filtered


def save_results(df_result, filename):
    """保存验证结果到 CSV 文件"""
    output_file = f'data/{filename}_salts_with_cas.csv'
    df_result.to_csv(output_file, index=False)
    print(f"✓ 结果已保存至: {output_file}")
    print(f"✓ 最终保留: {len(df_result)} 个季铵阳离子\n")


def main():
    """主流程: 加载数据、验证盐、添加CAS列并保存结果"""
    filename = 'r4n_c20_validated_pubchempy'
    
    # 加载数据
    input_file = f'data/{filename}.csv'
    df = pd.read_csv(input_file)
    print(f"已加载: {input_file} ({len(df)} 个化合物)")
    
    # 验证盐并添加CAS列
    df_result = validate_salts_and_add_cas_columns(df, filename)
    
    # 保存结果
    save_results(df_result, filename)
    
    print("="*80)
    print("处理完成!")
    print("="*80)
    
if __name__ == "__main__":
    main()
    import gc
    gc.collect()