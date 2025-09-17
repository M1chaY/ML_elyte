#!/usr/bin/env python3
# 测试修复后的数据提取功能

import sys
sys.path.append('..')
from data import QM9DataExtractor, safe_float_convert

# 测试安全转换函数
print("=== 测试安全数值转换函数 ===")
test_values = [
    '1.23e-4',          # 标准科学计数法
    '2.1997*^-6',       # Mathematica风格
    '-1.5*^3',          # 负数指数
    '3.14159',          # 普通小数
    '42',               # 整数
    42,                 # 数值类型
    3.14,               # 浮点数
]

for val in test_values:
    result = safe_float_convert(val)
    print(f"输入: {val} -> 输出: {result}")

print("\n=== 测试数据提取 ===")
try:
    extractor = QM9DataExtractor('../data/demo/qm9_demo.csv')
    info = extractor.get_molecule_info('C')
    
    coords_data = info['coordinates_data']
    props_data = info['properties_data']
    
    print("数据提取成功!")
    print(f"分子SMILES: {coords_data['smiles']}")
    print(f"原子数量: {coords_data['num_atoms']}")
    print(f"坐标形状: {coords_data['coordinates'].shape}")
    print(f"电荷形状: {coords_data['charges'].shape}")
    print(f"属性数量: {len(props_data['properties'])}")
    
    print("\n坐标样例:")
    for i, (atom, coord, charge) in enumerate(zip(coords_data['atoms'], 
                                                  coords_data['coordinates'], 
                                                  coords_data['charges'])):
        print(f"  原子{i}: {atom}, 坐标: {coord}, 电荷: {charge}")
    
    print("\n属性样例:")
    for prop, value in list(props_data['properties'].items())[:5]:
        print(f"  {prop}: {value}")
        
except Exception as e:
    print(f"测试失败: {e}")
    import traceback
    traceback.print_exc()
