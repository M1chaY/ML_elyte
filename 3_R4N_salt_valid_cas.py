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
    
def validate_and_get_cas(smiles: str, verbose: bool = False, idx: Optional[int] = None, total: Optional[int] = None):
    """
    验证 SMILES 并从 PubChem 获取 CAS 号
    
    参数:
        smiles: str - SMILES 字符串
        verbose: bool - 是否打印进度
        idx: int - 当前序号(可选)
        total: int - 总数(可选)
    
    返回:
        dict: 包含 SMILES、CAS 号等信息
    """
    prefix = f"[{idx}/{total}] " if (idx is not None and total is not None) else ""
   
    try:
        if verbose and idx % 10 == 1:  # 每10个打印一次
            print(f"{prefix}Querying...", flush=True)
       
        results = pcp.get_compounds(smiles, 'smiles')
       
        if results:
            compound = results[0]
            synonyms = compound.synonyms if compound.synonyms else []
           
            # 筛选 CAS 号格式 (xxx-xx-x)
            cas_numbers = [s for s in synonyms if '-' in s and all(part.isdigit() for part in s.split('-'))]
           
            if cas_numbers:
                cas_str = ', '.join(cas_numbers)
                return {
                    'smiles': smiles,
                    'exists': True,
                    'cid': compound.cid,
                    'name': compound.iupac_name,
                    'molecular_weight': compound.molecular_weight,
                    'cas': cas_str
                }
       
        return {'smiles': smiles, 'exists': False, 'cas': ''}
       
    except Exception as e:
        return {'smiles': smiles, 'exists': False, 'cas': '', 'error': str(e)}

def create_salt_dataframes(df, filename):
    """创建溴化盐和氯化盐数据框并合并"""
    smiles = df['SMILES']
    
    # 生成盐的 SMILES
    df['SMILES_Bromide'] = smiles.apply(lambda x: add_halide_to_smiles(x, 'Br'))
    df['SMILES_Chloride'] = smiles.apply(lambda x: add_halide_to_smiles(x, 'Cl'))

    # 创建溴化盐数据框
    df_bromide = df[['Index', 'Num_c', 'SMILES', 'SMILES_Bromide']].copy()
    df_bromide['Halide_Type'] = 'Bromide'
    df_bromide['SMILES_Salt'] = df_bromide['SMILES_Bromide']
    df_bromide['Original_Index'] = df_bromide['Index']
    df_bromide = df_bromide[['Original_Index', 'Num_c', 'SMILES', 'SMILES_Salt', 'Halide_Type']]

    # 创建氯化盐数据框
    df_chloride = df[['Index', 'Num_c', 'SMILES', 'SMILES_Chloride']].copy()
    df_chloride['Halide_Type'] = 'Chloride'
    df_chloride['SMILES_Salt'] = df_chloride['SMILES_Chloride']
    df_chloride['Original_Index'] = df_chloride['Index']
    df_chloride = df_chloride[['Original_Index', 'Num_c', 'SMILES', 'SMILES_Salt', 'Halide_Type']]

    # 合并数据框
    df_combined = pd.concat([df_bromide, df_chloride], ignore_index=True)
    df_combined = df_combined.sort_values(by=['Original_Index', 'Halide_Type']).reset_index(drop=True)
    df_combined.insert(0, 'Index', range(1, len(df_combined) + 1))

    # 保存合并后的数据
    output_filename = f'data/{filename}_salts_combined.csv'
    df_combined.to_csv(output_filename, index=False)
    print(f"Combined salts saved: {output_filename} (n={len(df_combined)})")
    
    return df_combined


def validate_salts_with_cas(df_combined, filename):
    """验证盐并从 PubChem 获取 CAS 号"""
    smiles_list = df_combined['SMILES_Salt'].tolist()
    total = len(smiles_list)
    
    print(f"Validating {total} compounds...")
    
    # 查询 CAS 号
    results = []
    for i, smiles in enumerate(smiles_list, start=1):
        result = validate_and_get_cas(smiles, verbose=True, idx=i, total=total)
        results.append(result)

    # 将结果添加到数据框
    results_df = pd.DataFrame(results)
    df_combined['CAS'] = results_df['cas'].tolist()

    # 筛选出有 CAS 号的化合物
    df_with_cas = df_combined[df_combined['CAS'] != ''].copy()
    success_rate = len(df_with_cas) / len(df_combined) * 100

    print(f"Validation complete: {len(df_with_cas)}/{len(df_combined)} ({success_rate:.1f}%)")
    
    return df_combined, df_with_cas


def save_results(df_combined, df_with_cas, filename):
    """保存验证结果到 CSV 文件"""
    # 保存有 CAS 号的化合物
    output_validated = f'data/{filename}_salts_with_cas.csv'
    df_with_cas.to_csv(output_validated, index=False)
    print(f"Saved with CAS: {output_validated}")

    # 保存所有化合物
    output_all = f'data/{filename}_salts_all_with_cas_info.csv'
    df_combined.to_csv(output_all, index=False)
    print(f"Saved all: {output_all}")


def main():
    """主流程: 加载数据、创建盐、验证并保存结果"""
    filename = 'r4n_smiles_c20'
    
    # 加载数据
    df = pd.read_csv(f'data/{filename}_pubchem_validated.csv')
    print(f"Loaded: {len(df)} compounds")
    
    # 创建盐数据框
    df_combined = create_salt_dataframes(df, filename)
    
    # 验证并获取 CAS 号
    df_combined, df_with_cas = validate_salts_with_cas(df_combined, filename)
    
    # 保存结果
    save_results(df_combined, df_with_cas, filename)
    
if __name__ == "__main__":
    main()
    import gc
    gc.collect()