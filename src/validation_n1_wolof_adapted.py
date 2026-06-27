# src/validation_n1_wolof_adapted.py

import json
from pathlib import Path
from typing import List, Dict
from collections import Counter
import re


class WolofValidator:
    """Validation N1 adaptée au wolof (sans fastText)"""
    
    def __init__(self):
        self.stats = {
            'total': 0,
            'passed': 0,
            'failed_length': 0,
            'failed_encoding': 0,
            'failed_html': 0,
            'failed_wolof_chars': 0
        }
        
        # Caractères typiques du wolof
        self.wolof_chars = set('ëñóàéèɛɔŋ')
        
        # Mots wolof courants (pour validation supplémentaire)
        self.wolof_words = {
            'la', 'na', 'ci', 'mu', 'du', 'di', 'ko', 'nga', 'naa',
            'ak', 'gi', 'yi', 'bi', 'wi', 'mi', 'si', 'ki',
            'mooy', 'doon', 'ñu', 'ñuy', 'muy', 'duy',
            'sama', 'sa', 'man', 'yow', 'ñoom',
            'jëf', 'wax', 'def', 'dem', 'jóg', 'réew', 'nit',
            'boppam', 'mbir', 'yàlla', 'allaaxiraa'
        }
    
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
        html_tags = re.findall(r'<[^>]+>', text)
        
        if html_tags:
            html_chars = sum(len(tag) for tag in html_tags)
            text_chars = len(text)
            if html_chars / text_chars > 0.1:
                return False
        
        return True
    
    def detect_wolof(self, text: str) -> Dict:
        """
        Détecter si le texte est en wolof par heuristique
        
        Critères :
        1. Présence de caractères typiques wolof (ë, ñ, ó, etc.)
        2. Présence de mots wolof courants
        3. Ratio caractères spéciaux / texte total
        
        Returns:
            {'is_wolof': bool, 'confidence': float, 'evidence': dict}
        """
        text_lower = text.lower()
        
        # Critère 1 : Caractères typiques wolof
        wolof_char_count = sum(1 for char in text if char in self.wolof_chars)
        has_wolof_chars = wolof_char_count > 0
        
        # Critère 2 : Mots wolof courants
        words = re.findall(r'\b\w+\b', text_lower)
        wolof_word_matches = [w for w in words if w in self.wolof_words]
        wolof_word_ratio = len(wolof_word_matches) / len(words) if words else 0
        
        # Critère 3 : Densité caractères spéciaux
        char_density = wolof_char_count / len(text) if text else 0
        
        # Décision : au moins UN critère doit être rempli
        is_wolof = (
            has_wolof_chars or  # Au moins 1 caractère spécial wolof
            wolof_word_ratio >= 0.1  # Au moins 10% de mots wolof courants
        )
        
        # Confiance (0-1)
        confidence = min(1.0, (
            (0.4 if has_wolof_chars else 0) +
            (wolof_word_ratio * 0.6)
        ))
        
        evidence = {
            'wolof_char_count': wolof_char_count,
            'wolof_word_count': len(wolof_word_matches),
            'wolof_word_ratio': wolof_word_ratio,
            'char_density': char_density,
            'has_wolof_chars': has_wolof_chars
        }
        
        return {
            'is_wolof': is_wolof,
            'confidence': confidence,
            'evidence': evidence
        }
    
    def validate_article(self, article: Dict) -> Dict:
        """Valider un article selon les critères adaptés"""
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
        
        # Test 4 : Détection wolof
        wolof_detection = self.detect_wolof(text)
        
        if wolof_detection['is_wolof']:
            article['validation_n1'] = {
                'passed': True,
                'wolof_confidence': wolof_detection['confidence'],
                'wolof_evidence': wolof_detection['evidence'],
                'word_count': len(text.split())
            }
            self.stats['passed'] += 1
        else:
            article['validation_n1'] = {
                'passed': False,
                'reason': 'not_wolof',
                'wolof_detection': wolof_detection
            }
            self.stats['failed_wolof_chars'] += 1
        
        return article
    
    def validate_corpus(self, articles: List[Dict]) -> List[Dict]:
        """Valider tout le corpus"""
        print("🔍 VALIDATION N1 : Filtrage Adapté Wolof")
        print("="*60)
        print("Critères:")
        print("  ✓ Longueur ≥20 tokens")
        print("  ✓ Encodage UTF-8 valide")
        print("  ✓ Pas de HTML résiduel")
        print("  ✓ Détection wolof (chars + mots)")
        print()
        
        validated_articles = []
        
        for idx, article in enumerate(articles, 1):
            validated = self.validate_article(article)
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
        print(f"  - Pas wolof: {self.stats['failed_wolof_chars']}")
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
            
            # Distribution confiance wolof
            confidences = [a.get('validation_n1', {}).get('wolof_confidence', 0) for a in passed]
            avg_conf = sum(confidences) / len(confidences)
            min_conf = min(confidences)
            max_conf = max(confidences)
            
            print(f"\n📊 Confiance détection wolof:")
            print(f"  - Moyenne: {avg_conf*100:.1f}%")
            print(f"  - Min: {min_conf*100:.1f}%")
            print(f"  - Max: {max_conf*100:.1f}%")
            
            # Caractères wolof détectés
            wolof_char_counts = [
                a.get('validation_n1', {}).get('wolof_evidence', {}).get('wolof_char_count', 0) 
                for a in passed
            ]
            avg_chars = sum(wolof_char_counts) / len(wolof_char_counts)
            
            print(f"\n🔤 Caractères wolof (ë, ñ, ó, etc.):")
            print(f"  - Moyenne par article: {avg_chars:.1f}")
            print(f"  - Total: {sum(wolof_char_counts)}")
    
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
        
        # Sauvegarder rejetés
        rejected_path = Path(rejected_file)
        rejected_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rejected_path, 'w', encoding='utf-8') as f:
            for article in rejected:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
        
        print(f"\n💾 Fichiers sauvegardés:")
        print(f"  - Validés: {passed_file} ({len(passed)} articles)")
        print(f"  - Rejetés: {rejected_file} ({len(rejected)} articles)")


def main():
    """Script principal - Validation N1 Wolof"""
    print("="*60)
    print("🚀 VALIDATION N1 : WOLOF ADAPTÉ")
    print("="*60 + "\n")
    
    # Initialiser le validateur
    validator = WolofValidator()
    
    # Charger le corpus
    input_file = "data/final/wolof_online_text_corpus.jsonl"
    print(f"📂 Chargement: {input_file}\n")
    
    articles = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            articles.append(json.loads(line))
    
    print(f"✅ {len(articles)} articles chargés\n")
    
    # Valider le corpus
    validated_articles = validator.validate_corpus(articles)
    
    # Générer rapport
    validator.generate_report(validated_articles)
    
    # Sauvegarder résultats
    passed_file = "data/validated/n1_wolof_passed.jsonl"
    rejected_file = "data/validated/n1_wolof_rejected.jsonl"
    validator.save_results(validated_articles, passed_file, rejected_file)
    
    print("\n" + "="*60)
    print("✅ VALIDATION N1 TERMINÉE")
    print("="*60)
    print("\n💡 Prochaine étape:")
    print("  → Validation N2 : Déduplication avec corpus Alwaly (40 MB)")


if __name__ == "__main__":
    main()
