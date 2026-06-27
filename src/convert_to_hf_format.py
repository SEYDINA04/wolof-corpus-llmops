# convert_to_hf_format.py

import json
import re
from pathlib import Path
from typing import List, Dict


class CorpusConverter:
    """Convertir corpus wolof-online vers format HuggingFace"""

    def __init__(self, split_sentences: bool = True):
        """
        Args:
            split_sentences: Si True, splitter en phrases. Si False, garder articles complets.
        """
        self.split_sentences = split_sentences
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')

    def split_text_to_sentences(self, text: str) -> List[str]:
        """Splitter un texte en phrases (sépare sur . ! ?)"""
        # Normaliser espaces
        text_clean = text.replace('\n', ' ')
        text_clean = re.sub(r'\s+', ' ', text_clean).strip()

        # Splitter
        sentences = self.sentence_pattern.split(text_clean)

        # Filtrer phrases trop courtes
        sentences_clean = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent.split()) >= 3:  # Au moins 3 mots
                sentences_clean.append(sent)

        return sentences_clean

    def convert_article(self, article: dict) -> List[Dict]:
        """
        Convertir un article vers format HF

        Returns:
            Liste de dictionnaires au format HF
        """
        text = article.get('full_content', '')

        if not text:
            return []

        results = []

        if self.split_sentences:
            # Splitter en phrases
            sentences = self.split_text_to_sentences(text)

            for sent in sentences:
                results.append({
                    'text': sent,
                    'sources': ['wolof-online.com']
                })
        else:
            # Garder article complet
            results.append({
                'text': text,
                'sources': ['wolof-online.com']
            })

        return results

    def convert_corpus(self, input_file: str, output_file: str):
        """Convertir tout le corpus"""
        print("="*60)
        print("🔄 CONVERSION VERS FORMAT HUGGINGFACE")
        print("="*60)

        mode = "phrases" if self.split_sentences else "articles complets"
        print(f"\n📋 Mode: {mode}")
        print(f"📂 Input: {input_file}")
        print(f"📂 Output: {output_file}\n")

        # Charger corpus wolof-online
        print("📥 Chargement corpus wolof-online...")
        articles = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                articles.append(json.loads(line))

        print(f"✅ {len(articles)} articles chargés\n")

        # Convertir
        print("🔄 Conversion en cours...")
        all_items = []

        for idx, article in enumerate(articles, 1):
            items = self.convert_article(article)
            all_items.extend(items)

            if idx % 10 == 0:
                print(f"  Progression: {idx}/{len(articles)} articles | {len(all_items)} items générés")

        print(f"\n✅ {len(all_items)} items générés")

        # Sauvegarder
        print(f"\n💾 Sauvegarde: {output_file}")
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in all_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        # Statistiques
        total_words = sum(len(item['text'].split()) for item in all_items)
        avg_words = total_words / len(all_items)
        total_chars = sum(len(item['text']) for item in all_items)

        print("\n" + "="*60)
        print("📊 STATISTIQUES FINALES")
        print("="*60)
        print(f"  - Items générés: {len(all_items):,}")
        print(f"  - Total mots: {total_words:,}")
        print(f"  - Moyenne: {avg_words:.1f} mots/item")
        print(f"  - Total caractères: {total_chars:,}")

        if self.split_sentences:
            lengths = [len(item['text'].split()) for item in all_items]
            print(f"\n📏 Distribution longueur phrases:")
            print(f"  - Min: {min(lengths)} mots")
            print(f"  - Max: {max(lengths)} mots")
            print(f"  - Médiane: {sorted(lengths)[len(lengths)//2]} mots")

        # Échantillon
        print("\n" + "="*60)
        print("📝 ÉCHANTILLON (Format HuggingFace)")
        print("="*60)
        for i, item in enumerate(all_items[:3], 1):
            print(f"\n{i}. {item}")

        return all_items


def main():
    """Script principal"""
    print("="*60)
    print("🚀 CONVERSION CORPUS WOLOF-ONLINE")
    print("="*60 + "\n")

    print("Choisissez le mode de conversion:")
    print("  1. Splitter en phrases (recommandé)")
    print("  2. Garder articles complets")

    # Mode par défaut: splitter
    mode = input("\nVotre choix (1 ou 2) [défaut: 1]: ").strip() or "1"

    split_sentences = (mode == "1")

    # Initialiser convertisseur
    converter = CorpusConverter(split_sentences=split_sentences)

    # Chemins
    input_file = "data/validated/n1_wolof_passed.jsonl"
    output_file = "data/final/wolof_online_hf_format.jsonl"

    # Convertir
    items = converter.convert_corpus(input_file, output_file)

    print("\n" + "="*60)
    print("✅ CONVERSION TERMINÉE")
    print("="*60)
    print(f"\n📁 Fichier généré: {output_file}")
    print(f"📊 {len(items)} items au format HuggingFace")
    print("\n💡 Prochaine étape:")
    print("  → Merger avec le dataset HuggingFace")


if __name__ == "__main__":
    main()
