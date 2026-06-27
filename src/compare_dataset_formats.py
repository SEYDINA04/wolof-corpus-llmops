# compare_dataset_formats.py

import json
import pandas as pd
from pathlib import Path


def analyze_hf_dataset(parquet_path: str):
    """Analyser le format du dataset HuggingFace"""
    print("="*60)
    print("📊 DATASET HUGGINGFACE (existant)")
    print("="*60)
    
    # Charger le Parquet
    df = pd.read_parquet(parquet_path)
    
    print(f"\n📈 Statistiques générales:")
    print(f"  - Nombre d'exemples: {len(df):,}")
    print(f"  - Colonnes: {list(df.columns)}")
    print(f"  - Types: {dict(df.dtypes)}")
    
    print(f"\n🔍 Structure des données:")
    
    # Analyser la colonne 'text'
    if 'text' in df.columns:
        print(f"\n  Colonne 'text':")
        print(f"    - Type: {df['text'].dtype}")
        print(f"    - Exemples non-null: {df['text'].notna().sum()}")
        print(f"    - Longueur moyenne: {df['text'].str.len().mean():.0f} caractères")
        print(f"    - Exemple:")
        print(f"      '{df['text'].iloc[0][:100]}...'")
    
    # Analyser la colonne 'sources'
    if 'sources' in df.columns:
        print(f"\n  Colonne 'sources':")
        print(f"    - Type: {df['sources'].dtype}")
        
        # Vérifier si c'est une liste
        first_source = df['sources'].iloc[0]
        print(f"    - Format: {type(first_source)}")
        print(f"    - Exemple: {first_source}")
        
        # Si c'est une liste, analyser les sources uniques
        if isinstance(first_source, list):
            all_sources = df['sources'].explode().unique()
            print(f"    - Sources uniques: {list(all_sources)[:10]}")
    
    # Échantillon complet
    print(f"\n📝 Échantillon (1er exemple):")
    print(df.iloc[0].to_dict())
    
    return df


def analyze_wolof_online(jsonl_path: str):
    """Analyser le format du corpus wolof-online"""
    print("\n" + "="*60)
    print("📊 CORPUS WOLOF-ONLINE (à ajouter)")
    print("="*60)
    
    # Charger le JSONL
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"\n📈 Statistiques générales:")
    print(f"  - Nombre d'exemples: {len(data):,}")
    
    # Analyser les champs
    if data:
        first_item = data[0]
        print(f"  - Champs: {list(first_item.keys())}")
        
        print(f"\n🔍 Structure des données:")
        
        for key, value in first_item.items():
            print(f"\n  Champ '{key}':")
            print(f"    - Type: {type(value)}")
            
            if isinstance(value, str):
                print(f"    - Longueur: {len(value)} caractères")
                print(f"    - Aperçu: '{value[:100]}...'")
            elif isinstance(value, list):
                print(f"    - Longueur: {len(value)} éléments")
                print(f"    - Contenu: {value}")
            elif isinstance(value, dict):
                print(f"    - Clés: {list(value.keys())}")
            else:
                print(f"    - Valeur: {value}")
        
        print(f"\n📝 Échantillon (1er exemple):")
        print(json.dumps(first_item, indent=2, ensure_ascii=False)[:500] + "...")
    
    return data


def compare_formats(hf_df: pd.DataFrame, wolof_data: list):
    """Comparer et identifier les différences"""
    print("\n" + "="*60)
    print("🔍 COMPARAISON DES FORMATS")
    print("="*60)
    
    # Champs HF
    hf_cols = set(hf_df.columns)
    
    # Champs wolof-online
    wolof_cols = set(wolof_data[0].keys()) if wolof_data else set()
    
    print(f"\n📋 Champs du dataset HuggingFace:")
    for col in sorted(hf_cols):
        print(f"  ✓ {col}")
    
    print(f"\n📋 Champs du corpus wolof-online:")
    for col in sorted(wolof_cols):
        print(f"  ✓ {col}")
    
    # Différences
    print(f"\n🔄 Analyse des différences:")
    
    common = hf_cols & wolof_cols
    hf_only = hf_cols - wolof_cols
    wolof_only = wolof_cols - hf_cols
    
    if common:
        print(f"\n  ✅ Champs communs: {sorted(common)}")
    
    if hf_only:
        print(f"\n  ⚠️ Champs uniquement dans HF: {sorted(hf_only)}")
        print(f"     → Ces champs seront absents dans le corpus wolof-online")
    
    if wolof_only:
        print(f"\n  ⚠️ Champs uniquement dans wolof-online: {sorted(wolof_only)}")
        print(f"     → Ces champs seront ajoutés ou ignorés selon le besoin")
    
    # Vérifier compatibilité format 'sources'
    print(f"\n🔍 Vérification format 'sources':")
    
    if 'sources' in hf_cols and 'sources' in wolof_cols:
        hf_source = hf_df['sources'].iloc[0]
        wolof_source = wolof_data[0]['sources']
        
        print(f"  - HF format: {type(hf_source)} = {hf_source}")
        print(f"  - Wolof format: {type(wolof_source)} = {wolof_source}")
        
        if type(hf_source) == type(wolof_source):
            print(f"  ✅ Formats compatibles")
        else:
            print(f"  ⚠️ Formats différents - conversion nécessaire")


def generate_mapping_script(hf_cols: set, wolof_cols: set):
    """Générer un script de mapping"""
    print("\n" + "="*60)
    print("📝 SCRIPT DE MAPPING SUGGÉRÉ")
    print("="*60)
    
    print("""
def map_wolof_to_hf_format(wolof_item: dict) -> dict:
    \"\"\"Convertir format wolof-online vers format HF\"\"\"
    return {
        'text': wolof_item.get('full_content', ''),  # ← ADAPTER selon besoin
        'sources': wolof_item.get('sources', ['wolof-online.com'])
        # Ajouter autres champs si nécessaire
    }
    """)


def main():
    """Script principal"""
    print("="*60)
    print("🔍 COMPARAISON FORMATS DATASETS")
    print("="*60 + "\n")
    
    # Chemins
    hf_parquet = "wolof_centalized_corpus/data/train-00000-of-00001.parquet"
    wolof_jsonl = "data/validated/n1_wolof_passed.jsonl"
    
    # Vérifier existence
    if not Path(hf_parquet).exists():
        print(f"❌ Fichier HF introuvable: {hf_parquet}")
        return
    
    if not Path(wolof_jsonl).exists():
        print(f"❌ Fichier wolof-online introuvable: {wolof_jsonl}")
        return
    
    # Analyser les deux datasets
    hf_df = analyze_hf_dataset(hf_parquet)
    wolof_data = analyze_wolof_online(wolof_jsonl)
    
    # Comparer
    compare_formats(hf_df, wolof_data)
    
    # Générer mapping
    generate_mapping_script(set(hf_df.columns), set(wolof_data[0].keys()))
    
    print("\n" + "="*60)
    print("✅ ANALYSE TERMINÉE")
    print("="*60)
    print("\n💡 Actions recommandées:")
    print("  1. Vérifier les champs communs")
    print("  2. Décider quoi faire des champs supplémentaires")
    print("  3. Créer script de conversion si nécessaire")
    print("  4. Tester sur 5 exemples avant merge complet")


if __name__ == "__main__":
    main()
