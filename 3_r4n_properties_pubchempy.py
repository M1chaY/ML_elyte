import pubchempy as pcp
import pandas as pd

def get_ion_properties(cid_list, properties=None):
    """
    使用 get_properties 批量获取离子的性质
    
    参数:
        cid_list: CID列表
        properties: 要获取的属性列表,如果为None则获取常用属性
    
    返回:
        包含所有化合物信息的DataFrame
    """
    if properties is None:
        # 默认获取的属性 - 使用正确的属性名称
        properties = [
            'MolecularFormula',
            'MolecularWeight',
            'IUPACName',
            'Complexity',
            'RotatableBondCount',
            'HeavyAtomCount',
            'XLogP',  # 分配系数
        ]
    
    try:
        # 批量获取属性
        results = pcp.get_properties(properties, cid_list, namespace='cid')
        
        # 转换为DataFrame
        df_results = pd.DataFrame(results)
        
        return df_results
    
    except Exception as e:
        print(f"Error: {e}")
        return None
    
def main():
    # 读取数据
    df = pd.read_csv('data/r4n_c20_validated_cid.csv')
    df = df.drop(columns=['name','molecular_weight'])

    # 获取所有CID
    cid_list = df['cid'].astype(int).tolist()

    # 批量获取属性
    print(f"Searching {len(cid_list)} compounds for properties...")
    result_all = get_ion_properties(cid_list)

    if result_all is not None:
        # 转换数值列为正确的类型
        numeric_cols = ['MolecularWeight', 'Volume3D', 'EffectiveRotorCount3D', 
                        'Charge', 'XLogP', 'Complexity']
        for col in numeric_cols:
            if col in result_all.columns:
                result_all[col] = pd.to_numeric(result_all[col], errors='coerce')

        print(f"Successfully retrieved properties for {len(result_all)} compounds\n")
    
    # 把获取到的性质合并到原始DataFrame
    df = df.merge(result_all, left_on='cid', right_on='CID', how='left')
    df = df.drop(columns=['cid'])

    # 保存结果
    df.to_csv('data/r4n_c20_validated_pubchempy.csv', index=False)
    
if __name__ == "__main__":
    main()
    import gc
    gc.collect()
    