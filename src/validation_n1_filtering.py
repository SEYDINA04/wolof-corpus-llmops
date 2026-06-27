# src/validation_n1_filtering.py

import fasttext
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import re


class Level1Validator:
    """Validation Niveau 1 : Filtrage automatique"""

    def __init__(self, fasttext_model_path: str = "lid.176.bin"):
        """Initialiser avec le modèle fastText"""
        print("🔧 Chargement modèle fastText...")
        self.model = fasttext.load_model(fasttext_model_path)
        print("✅ Modèle chargé\n")

        self.stats = {
            'total': 0,
            'passed': 0,
            'failed_language': 0,
            'failed_length': 0,
            'failed_encoding': 0,
            'failed_html': 0
        }

    def detect_language(self, text: str, threshold: float = 0.7) -> Tuple[str, float]:
        """
        Détecter la langue avec fastText

        Returns:
            (langue, probabilité)
        """
        # Nettoyer le texte pour améliorer la détection
        text_clean = text.replace('\n', ' ').strip()

        if not text_clean:
            return ('unknown', 0.0)

        # Prédiction fastText
        predictions = self.model.predict(text_clean, k=1)

        # Extraire langue et probabilité
        lang_label = predictions[0][0].replace('__label__', '')
        prob = predictions[1][0]

        return (lang_label, prob)

    def check_length(self, text: str, min_tokens: int = 20) -> bool:
        """Vérifier longueur minimale (≥20 tokens)"""
        tokens = text.split()
        return len(tokens) >= min_tokens

    def check_encoding(self, text: str) -> bool:
        """Vérifier encodage UTF-8 valide"""
        try:
            text.encode('utf-8').decode('utf-8')
            return True
        except (UnicodeDecodeError, UnicodeEncodeError):
            return False

    def check_html_residual(self, text: str) -> bool:
        """Vérifier absence de HTML résiduel important"""
        # Compter les balises HTML
        html_tags = re.findall(r'<[^>]+>', text)

        # Si plus de 10% du texte est du HTML, rejeter
        if html_tags:
            html_chars = sum(len(tag) for tag in html_tags)
            text_chars = len(text)
            if html_chars / text_chars > 0.1:
                return False

        return True

    def validate_article(self, article: Dict, threshold: float = 0.7) -> Dict:
        """
        Valider un article selon les critères N1

        Returns:
            Article enrichi avec résultats de validation
        """
        self.stats['total'] += 1

        # Extraire le texte
        text = article.get('full_content', '') or article.get('text', '')

        if not text:
            article['validation_n1'] = {
                'passed': False,
                'reason': 'no_content'
            }
            self.stats['failed_length'] += 1
            return article

        # Test 1 : Encodage UTF-8
        if not self.check_encoding(text):
            article['validation_n1'] = {
                'passed': False,
                'reason': 'invalid_encoding'
            }
            self.stats['failed_encoding'] += 1
            return article

        # Test 2 : Longueur minimale
        if not self.check_length(text, min_tokens=20):
            article['validation_n1'] = {
                'passed': False,
                'reason': 'too_short',
                'word_count': len(text.split())
            }
            self.stats['failed_length'] += 1
            return article

        # Test 3 : HTML résiduel
        if not self.check_html_residual(text):
            article['validation_n1'] = {
                'passed': False,
                'reason': 'html_residual'
            }
            self.stats['failed_html'] += 1
            return article

        # Test 4 : Détection langue (wolof ≥70%)
        lang, prob = self.detect_language(text)

        if lang == 'wo' and prob >= threshold:
            article['validation_n1'] = {
                'passed': True,
                'language': lang,
                'language_prob': float(prob),
                'word_count': len(text.split())
            }
            self.stats['passed'] += 1
        else:
            article['validation_n1'] = {
                'passed': False,
                'reason': 'language_threshold',
                'detected_language': lang,
                'language_prob': float(prob),
                'threshold_required': threshold
            }
            self.stats['failed_language'] += 1

        return article

    def validate_corpus(self, articles: List[Dict], threshold: float = 0.7) -> List[Dict]:
        """Valider tout le corpus"""
        print("🔍 VALIDATION NIVEAU 1 : Filtrage Automatique")
        print("="*60)
        print(f"Seuil wolof: ≥{threshold*100:.0f}%")
        print(f"Longueur minimale: ≥20 tokens\n")

        validated_articles = []

        for idx, article in enumerate(articles, 1):
            validated = self.validate_article(article, threshold)
            validated_articles.append(validated)

            # Afficher progression
            if idx % 10 == 0 or idx == len(articles):
                passed_so_far = sum(1 for a in validated_articles
                                   if a.get('validation_n1', {}).get('passed'))
                print(f"📊 Progression: {idx}/{len(articles)} | "
                      f"✅ {passed_so_far} validés | "
                      f"❌ {idx - passed_so_far} rejetés")

        return validated_articles

    def generate_report(self, validated_articles: List[Dict]):
        """Générer rapport de validation"""
        print("\n" + "="*60)
        print("📊 RAPPORT VALIDATION N1")
        print("="*60)

        # Statistiques globales
        print(f"\n📈 Résultats:")
        print(f"  - Total articles: {self.stats['total']}")
        print(f"  - ✅ Validés: {self.stats['passed']} "
              f"({self.stats['passed']/self.stats['total']*100:.1f}%)")
        print(f"  - ❌ Rejetés: {self.stats['total'] - self.stats['passed']} "
              f"({(self.stats['total'] - self.stats['passed'])/self.stats['total']*100:.1f}%)")

        # Raisons de rejet
        print(f"\n🚫 Raisons de rejet:")
        print(f"  - Langue (<70% wolof): {self.stats['failed_language']}")
        print(f"  - Longueur (<20 tokens): {self.stats['failed_length']}")
        print(f"  - Encodage invalide: {self.stats['failed_encoding']}")
        print(f"  - HTML résiduel: {self.stats['failed_html']}")

        # Analyser les articles validés
        passed = [a for a in validated_articles if a.get('validation_n1', {}).get('passed')]

        if passed:
            total_words = sum(a.get('validation_n1', {}).get('word_count', 0) for a in passed)
            avg_words = total_words / len(passed)
            estimated_mb = (total_words * 7) / (1024 * 1024)

            print(f"\n✅ CORPUS VALIDÉ:")
            print(f"  - Articles: {len(passed)}")
            print(f"  - Total mots: {total_words:,}")
            print(f"  - Moyenne: {avg_words:.0f} mots/article")
            print(f"  - Volume: {estimated_mb:.2f} MB")

            # Distribution probabilité wolof
            probs = [a.get('validation_n1', {}).get('language_prob', 0) for a in passed]
            avg_prob = sum(probs) / len(probs)
            min_prob = min(probs)
            max_prob = max(probs)

            print(f"\n📊 Confiance détection wolof:")
            print(f"  - Moyenne: {avg_prob*100:.1f}%")
            print(f"  - Min: {min_prob*100:.1f}%")
            print(f"  - Max: {max_prob*100:.1f}%")

        # Analyser les rejets langue
        failed_lang = [a for a in validated_articles
                      if a.get('validation_n1', {}).get('reason') == 'language_threshold']

        if failed_lang:
            detected_langs = [a.get('validation_n1', {}).get('detected_language')
                            for a in failed_lang]
            lang_counts = Counter(detected_langs)

            print(f"\n🌍 Langues détectées (articles rejetés):")
            for lang, count in lang_counts.most_common(5):
                print(f"  - {lang}: {count} articles")

    def save_results(self, validated_articles: List[Dict],
                    passed_file: str, rejected_file: str):
        """Sauvegarder les résultats"""
        passed = [a for a in validated_articles
                 if a.get('validation_n1', {}).get('passed')]
        rejected = [a for a in validated_articles
                   if not a.get('validation_n1', {}).get('passed')]

        # Sauvegarder validés
        passed_path = Path(passed_file)
        passed_path.parent.mkdir(parents=True, exist_ok=True)
        with open(passed_path, 'w', encoding='utf-8') as f:
            for article in passed:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')

        # Sauvegarder rejetés (pour analyse)
        rejected_path = Path(rejected_file)
        rejected_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rejected_path, 'w', encoding='utf-8') as f:
            for article in rejected:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')

        print(f"\n💾 Fichiers sauvegardés:")
        print(f"  - Validés: {passed_file} ({len(passed)} articles)")
        print(f"  - Rejetés: {rejected_file} ({len(rejected)} articles)")


def main():
    """Script principal - Validation N1"""
    print("="*60)
    print("🚀 VALIDATION N1 : FILTRAGE AUTOMATIQUE")
    print("="*60 + "\n")

    # Initialiser le validateur
    validator = Level1Validator(fasttext_model_path="lid.176.bin")

    # Charger le corpus
    input_file = "data/final/wolof_online_text_corpus.jsonl"
    print(f"📂 Chargement: {input_file}\n")

    articles = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            articles.append(json.loads(line))

    print(f"✅ {len(articles)} articles chargés\n")

    # Valider le corpus
    validated_articles = validator.validate_corpus(articles, threshold=0.7)

    # Générer rapport
    validator.generate_report(validated_articles)

    # Sauvegarder résultats
    passed_file = "data/validated/n1_passed_corpus.jsonl"
    rejected_file = "data/validated/n1_rejected_corpus.jsonl"
    validator.save_results(validated_articles, passed_file, rejected_file)

    print("\n" + "="*60)
    print("✅ VALIDATION N1 TERMINÉE")
    print("="*60)
    print("\n💡 Prochaine étape:")
    print("  → Validation N2 : Déduplication (MinHash LSH)")


if __name__ == "__main__":
    main()
