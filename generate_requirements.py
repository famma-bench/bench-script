import os
import ast
from pathlib import Path

def get_imports_from_file(file_path):
    """Extract import statements from a Python file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            tree = ast.parse(file.read())
        except:
            return set()
    
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.add(name.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
    return imports

def generate_requirements():
    """Generate requirements.txt from all Python files in the project."""
    # 标准库列表
    stdlib_modules = set([
        'collections', 'argparse', 'pathlib', 'sys', 'os', 'ast', 
        'json', 'logging', 'datetime', 'time', 'random', 'math'
    ])
    
    # 本地模块（项目特定的模块）
    local_modules = set([
        'famma_runner', 'easyllm_kit', 'utils'
    ])
    
    # 遍历所有Python文件
    all_imports = set()
    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                imports = get_imports_from_file(file_path)
                all_imports.update(imports)
    
    # 过滤掉标准库和本地模块
    third_party_imports = all_imports - stdlib_modules - local_modules
    
    # 写入requirements.txt
    with open('requirements_auto.txt', 'w', encoding='utf-8') as f:
        f.write('# 自动生成的依赖列表\n')
        for package in sorted(third_party_imports):
            f.write(f'{package}\n')

if __name__ == '__main__':
    generate_requirements() 