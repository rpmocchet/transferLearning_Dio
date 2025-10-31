#!/usr/bin/env python
"""
Script para verificar a configuração do projeto.
"""

import os
import sys


def check_file_exists(filepath, description):
    """Verifica se um arquivo existe."""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists


def check_directory_exists(dirpath, description):
    """Verifica se um diretório existe."""
    exists = os.path.isdir(dirpath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {dirpath}")
    return exists


def verify_python_syntax(filepath):
    """Verifica sintaxe Python de um arquivo."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            compile(f.read(), filepath, 'exec')
        return True
    except SyntaxError as e:
        print(f"   ❌ Erro de sintaxe: {e}")
        return False


def main():
    print("=" * 60)
    print("🔍 VERIFICAÇÃO DO PROJETO - TRANSFER LEARNING")
    print("=" * 60)
    
    all_ok = True
    
    # Verificar estrutura de diretórios
    print("\n📁 Verificando estrutura de diretórios...")
    dirs = [
        ('src', 'Código fonte'),
        ('notebooks', 'Notebooks Jupyter'),
        ('models', 'Modelos'),
        ('data', 'Dados'),
        ('docs', 'Documentação'),
    ]
    
    for dirname, description in dirs:
        if not check_directory_exists(dirname, description):
            all_ok = False
    
    # Verificar arquivos principais
    print("\n📄 Verificando arquivos principais...")
    files = [
        ('README.md', 'README principal'),
        ('requirements.txt', 'Dependências'),
        ('.gitignore', 'Git ignore'),
        ('src/__init__.py', 'Init do pacote'),
        ('src/train_model.py', 'Script de treinamento'),
        ('src/predict.py', 'Script de predição'),
        ('src/utils.py', 'Utilitários'),
        ('notebooks/transfer_learning_demo.ipynb', 'Notebook demo'),
        ('docs/GUIDE.md', 'Guia completo'),
        ('data/README.md', 'README de dados'),
        ('models/README.md', 'README de modelos'),
    ]
    
    for filepath, description in files:
        if not check_file_exists(filepath, description):
            all_ok = False
    
    # Verificar sintaxe Python
    print("\n🐍 Verificando sintaxe dos scripts Python...")
    python_files = [
        'src/__init__.py',
        'src/train_model.py',
        'src/predict.py',
        'src/utils.py',
    ]
    
    for filepath in python_files:
        if os.path.exists(filepath):
            print(f"   Verificando {filepath}...", end=" ")
            if verify_python_syntax(filepath):
                print("✅")
            else:
                print("❌")
                all_ok = False
        else:
            print(f"   ❌ {filepath} não encontrado")
            all_ok = False
    
    # Verificar conteúdo do README
    print("\n📖 Verificando conteúdo do README...")
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            readme_content = f.read()
            required_sections = [
                'Transfer Learning',
                'Instalação',
                'Uso',
                'Estrutura do Projeto',
                'cats_vs_dogs',
            ]
            
            for section in required_sections:
                if section in readme_content:
                    print(f"   ✅ Seção encontrada: {section}")
                else:
                    print(f"   ⚠️  Seção não encontrada: {section}")
    except Exception as e:
        print(f"   ❌ Erro ao ler README: {e}")
        all_ok = False
    
    # Verificar requirements.txt
    print("\n📦 Verificando dependências...")
    try:
        with open('requirements.txt', 'r', encoding='utf-8') as f:
            requirements = f.read()
            required_packages = [
                'tensorflow',
                'tensorflow-datasets',
                'numpy',
                'matplotlib',
                'pillow',
                'jupyter',
            ]
            
            for package in required_packages:
                if package in requirements.lower():
                    print(f"   ✅ {package}")
                else:
                    print(f"   ⚠️  {package} não encontrado")
    except Exception as e:
        print(f"   ❌ Erro ao ler requirements.txt: {e}")
        all_ok = False
    
    # Resultado final
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ VERIFICAÇÃO CONCLUÍDA COM SUCESSO!")
        print("=" * 60)
        print("\n🚀 Próximos passos:")
        print("   1. Instale as dependências: pip install -r requirements.txt")
        print("   2. Execute o treinamento: python src/train_model.py")
        print("   3. Faça predições: python src/predict.py --image sua_imagem.jpg")
        print("   4. Explore o notebook: jupyter notebook notebooks/transfer_learning_demo.ipynb")
        return 0
    else:
        print("⚠️  VERIFICAÇÃO ENCONTROU PROBLEMAS")
        print("=" * 60)
        print("\nVerifique os itens marcados com ❌ acima.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
