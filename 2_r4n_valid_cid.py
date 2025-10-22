import pubchempy as pcp
from typing import Optional
import pandas as pd

def validate_against_pubchem(smiles, verbose: bool = False, idx: Optional[int] = None, total: Optional[int] = None):
    """验证生成的SMILES是否在PubChem中存在
    参数:
        smiles: str
        verbose: 是否实时打印查询进度与结果
        idx: 当前序号（可选）
        total: 总数（可选）
    返回:
        dict: 包含是否存在、CID、IUPAC名称和分子量等信息
    """
    prefix = f"[{idx}/{total}] " if (idx is not None and total is not None) else ""
    try:
        if verbose:
            print(f"{prefix}Searching PubChem: {smiles}", flush=True)
        results = pcp.get_compounds(smiles, 'smiles')
        if results:
            compound = results[0]
            info = {
                'smiles': smiles,
                'exists': True,
                'cid': compound.cid,
                'name': compound.iupac_name,
                'molecular_weight': compound.molecular_weight
            }
            if verbose:
                nm = info['name'] if info['name'] else '(No IUPAC Name)'
                print(f"{prefix}Found: CID={info['cid']}, Name={nm}, Molecular Weight={info['molecular_weight']}", flush=True)
            return info
    except Exception as e:
        if verbose:
            print(f"{prefix}Error occurred: {e}", flush=True)
        return {'smiles': smiles, 'exists': False, 'error': str(e)}
    # 未找到
    if verbose:
        print(f"{prefix}Not found", flush=True)
    return {'smiles': smiles, 'exists': False}

def main():
    # 读取SMILES列表和对应的元数据
    df = pd.read_csv(f'data/r4n_smiles_c20.csv')
    lists = df['SMILES'].tolist()

    # 实时输出：传入 idx 和 total，并开启 verbose
    results = [
        validate_against_pubchem(smiles, verbose=True, idx=i, total=len(lists))
        for i, smiles in enumerate(lists, start=1)
    ]
    results_df = pd.DataFrame(results)

    # df和results_df直接拼接
    final_df = pd.concat([df, results_df], axis=1)

    # 保存结果
    final_df = final_df.drop(columns=['smiles', 'exists'])

    # 删去cid为NaN的行
    final_df = final_df.dropna(subset=['cid'])

    # 保存结果
    final_df.to_csv(f'data/r4n_c20_validated_cid.csv', index=False)
    
if __name__ == "__main__":
    main()
    # 释放资源
    import gc
    gc.collect()