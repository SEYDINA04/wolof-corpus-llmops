# split_corpus_to_sentences.py

import json
import re
from pathlib import Path
from typing import List


class SentenceSplitter:
    """Split corpus en phrases individuelles"""
    
    def __init__(self):
        # Regex pour splitter sur . ! ?
        # Garde les séparateurs avec lookbehind
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    
    def split_text_to_sentences(self, text: str) -> List[str]:
        """
        Splitter un texte en phrases
        
        Sépare sur: . ! ?
        Nettoie: espaces multiples, lignes vides
        """
        # Normaliser les espaces et retours à la ligne
        text_clean = text.replace('\n', ' ')
        text_clean = re.sub(r'\s+', ' ', text_clean).strip()
        
        # Splitter sur . ! ?
        sentences = self.sentence_pattern.split(text_clean)
        
        # Nettoyer chaque phrase
        sentences_clean = []
        for sent in sentences:
            sent = sent.strip()
            
            # Filtrer phrases trop courtes (< 3 mots)
            if len(sent.split()) >= 3:
                sentences_clean.append(sent)
        
        return sentences_clean
    
    def process_corpus(self, input_file: str, output_file: str):
        """Traiter tout le corpus"""
        print("="*60)
        print("🚀 SPLITTING CORPUS EN PHRASES")
        print("="*60 + "\n")
        
        print(f"📂 Lecture: {input_file}\n")
        
        all_sentences = []
        articles_processed = 0
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                article = json.loads(line)
                text = article.get('full_content', '')
                
                if not text:
                    continue
                
                # Splitter en phrases
                sentences = self.split_text_to_sentences(text)
                
                # Ajouter avec métadonnées source
                for sent in sentences:
                    all_sentences.append({
                        'text': sent,
                        'sources': ['wolof-online.com'],
                        'article_id': article.get('id'),
                        'article_url': article.get('url')
                    })
                
                articles_processed += 1
                
                # Afficher progression
                if articles_processed % 10 == 0:
                    print(f"📊 Progression: {articles_processed} articles | "
                          f"{len(all_sentences)} phrases extraites")
        
        print(f"\n✅ {articles_processed} articles traités")
        print(f"✅ {len(all_sentences)} phrases extraites")
        
        # Statistiques
        total_words = sum(len(s['text'].split()) for s in all_sentences)
        avg_words = total_words / len(all_sentences)
        
        print("\n" + "="*60)
        print("📊 STATISTIQUES")
        print("="*60)
        print(f"  - Total phrases: {len(all_sentences):,}")
        print(f"  - Total mots: {total_words:,}")
        print(f"  - Moyenne: {avg_words:.1f} mots/phrase")
        
        # Distribution longueur phrases
        lengths = [len(s['text'].split()) for s in all_sentences]
        print(f"\n📏 Distribution longueur phrases:")
        print(f"  - Min: {min(lengths)} mots")
        print(f"  - Max: {max(lengths)} mots")
        print(f"  - Médiane: {sorted(lengths)[len(lengths)//2]} mots")
        
        # Sauvegarder
        print(f"\n💾 Sauvegarde: {output_file}")
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sent in all_sentences:
                f.write(json.dumps(sent, ensure_ascii=False) + '\n')
        
        print(f"✅ {len(all_sentences)} phrases sauvegardées")
        
        # Échantillon
        print("\n" + "="*60)
        print("📝 ÉCHANTILLON (5 premières phrases)")
        print("="*60)
        for i, sent in enumerate(all_sentences[:5], 1):
            print(f"\n{i}. {sent['text']}")


def main():
    """Script principal"""
    splitter = SentenceSplitter()
    
    input_file = "data/validated/n1_wolof_passed.jsonl"
    output_file = "data/final/wolof_online_sentences.jsonl"
    
    splitter.process_corpus(input_file, output_file)
    
    print("\n" + "="*60)
    print("✅ SPLITTING TERMINÉ")
    print("="*60)
    print("\n💡 Prochaine étape:")
    print("  1. Charger dataset HuggingFace")
    print("  2. Merger avec ce fichier")
    print("  3. Push vers HF")


if __name__ == "__main__":
    main()
