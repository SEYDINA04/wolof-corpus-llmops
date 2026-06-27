# merge_without_dedup.py

import pandas as pd
import json
from pathlib import Path
import numpy as np


def load_hf_dataset(parquet_path: str) -> pd.DataFrame:
    """Charger le dataset HuggingFace existant"""
    print("📂 Chargement dataset HuggingFace existant...")
    df = pd.read_parquet(parquet_path)
    print(f"✅ {len(df):,} items chargés")
    return df


def load_wolof_online(jsonl_path: str) -> pd.DataFrame:
    """Charger le corpus wolof-online converti"""
    print("\n📂 Chargement corpus wolof-online...")
    
    items = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line))
    
    # Convertir sources en numpy array (format HF)
    for item in items:
        item['sources'] = np.array(item['sources'], dtype=object)
    
    df = pd.DataFrame(items)
    print(f"✅ {len(df):,} items chargés")
    
    return df


def merge_datasets(hf_df: pd.DataFrame, wolof_df: pd.DataFrame) -> pd.DataFrame:
    """Merger les deux datasets SANS déduplication"""
    print("\n🔗 Fusion des datasets (SANS déduplication)...")
    
    print(f"  - Dataset HF: {len(hf_df):,} items")
    print(f"  - Wolof-online: {len(wolof_df):,} items")
    
    # Combiner sans déduplication
    combined_df = pd.concat([hf_df, wolof_df], ignore_index=True)
    
    print(f"  - Total final: {len(combined_df):,} items")
    
    return combined_df


def generate_statistics(df: pd.DataFrame):
    """Afficher statistiques finales"""
    print("\n" + "="*60)
    print("📊 STATISTIQUES DATASET FINAL")
    print("="*60)
    
    total_items = len(df)
    total_chars = df['text'].str.len().sum()
    avg_chars = df['text'].str.len().mean()
    total_words = df['text'].str.split().str.len().sum()
    avg_words = df['text'].str.split().str.len().mean()
    
    print(f"\n📈 Volume:")
    print(f"  - Total items: {total_items:,}")
    print(f"  - Total mots: {total_words:,}")
    print(f"  - Moyenne: {avg_words:.1f} mots/item")
    print(f"  - Total caractères: {total_chars:,}")
    print(f"  - Moyenne: {avg_chars:.1f} caractères/item")
    
    # Distribution par source
    print(f"\n📂 Distribution par source:")
    
    # Exploser les arrays sources
    all_sources = []
    for sources in df['sources']:
        if isinstance(sources, np.ndarray):
            all_sources.extend(sources.tolist())
        elif isinstance(sources, list):
            all_sources.extend(sources)
        else:
            all_sources.append(str(sources))
    
    from collections import Counter
    source_counts = Counter(all_sources)
    
    for source, count in source_counts.most_common():
        print(f"  - {source}: {count:,} items")


def save_parquet(df: pd.DataFrame, output_path: str):
    """Sauvegarder en Parquet"""
    print(f"\n💾 Sauvegarde Parquet: {output_path}")
    
    # S'assurer que le répertoire existe
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder
    df.to_parquet(output_path, index=False)
    
    # Vérifier taille
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"✅ Fichier sauvegardé ({size_mb:.1f} MB)")


def main():
    """Script principal"""
    print("="*60)
    print("🚀 MERGE SIMPLE (SANS DÉDUPLICATION)")
    print("="*60 + "\n")
    
    # Chemins
    hf_parquet = "wolof_centalized_corpus/data/train-00000-of-00001.parquet"
    wolof_jsonl = "data/final/wolof_asr_whosper_v5_HF_FORMAT.jsonl"
    output_parquet = "wolof_centalized_corpus/data/train-00000-of-00001.parquet"
    
    # Charger datasets
    hf_df = load_hf_dataset(hf_parquet)
    wolof_df = load_wolof_online(wolof_jsonl)
    
    # Merger SANS déduplication
    merged_df = merge_datasets(hf_df, wolof_df)
    
    # Statistiques
    generate_statistics(merged_df)
    
    # Sauvegarder
    save_parquet(merged_df, output_parquet)
    
    print("\n" + "="*60)
    print("✅ MERGE TERMINÉ")
    print("="*60)
    print("\n💡 Prochaines étapes:")
    print("  1. cd wolof_centalized_corpus")
    print("  2. git add data/train-00000-of-00001.parquet")
    print("  3. git commit -m 'Add wolof-online.com corpus (1,182 phrases, 35k words)'")
    print("  4. git push")


if __name__ == "__main__":
    main()
