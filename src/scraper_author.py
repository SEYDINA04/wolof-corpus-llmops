# src/scraper_author.py

import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import json
import time
from pathlib import Path


class WolofOnlineAuthorScraper:
    """Scraper de la page auteur pour récupérer tous les articles"""
    
    def __init__(self, base_url: str = "https://www.wolof-online.com"):
        self.base_url = base_url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Academic Research Bot) Wolof Corpus Collection'
        }
    
    def scrape_author_page(self, author_id: int = 1) -> List[Dict]:
        """
        Scraper tous les articles d'un auteur
        
        Args:
            author_id: ID de l'auteur (par défaut 1 = 'tam')
        
        Returns:
            List[Dict]: Liste de tous les articles de l'auteur
        """
        all_articles = []
        page_num = 1
        
        print(f"🔍 Scraping page auteur (ID: {author_id})...")
        print("="*60 + "\n")
        
        while True:
            # Construire l'URL
            if page_num == 1:
                url = f"{self.base_url}/?author={author_id}"
            else:
                url = f"{self.base_url}/?author={author_id}&paged={page_num}"
            
            try:
                # Requête HTTP
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                
                # Parser le HTML
                soup = BeautifulSoup(response.content, 'lxml')
                
                # Extraire les articles
                articles = soup.find_all('article', class_='posts-entry')
                
                # Si aucun article, fin de pagination
                if not articles:
                    print(f"   ⚠️ Aucun article sur page {page_num}, fin de pagination")
                    break
                
                print(f"📄 Page {page_num}: {len(articles)} articles trouvés")
                
                # Extraire métadonnées
                for article in articles:
                    article_data = self._extract_article_metadata(article)
                    if article_data:
                        all_articles.append(article_data)
                
                # Vérifier s'il y a une page suivante
                next_link = soup.find('a', class_='next page-numbers')
                if not next_link:
                    break
                
                page_num += 1
                time.sleep(1)  # Respecter le serveur
                
            except Exception as e:
                print(f"   ❌ Erreur page {page_num}: {e}")
                break
        
        print(f"\n🎯 Total: {len(all_articles)} articles collectés depuis page auteur")
        return all_articles
    
    def _extract_article_metadata(self, article) -> Dict:
        """Extraire les métadonnées d'un article"""
        try:
            # Titre
            title_tag = article.find('h2', class_='entry-title')
            title = title_tag.get_text(strip=True) if title_tag else None
            
            # URL
            link_tag = title_tag.find('a') if title_tag else None
            url = link_tag.get('href') if link_tag else None
            
            # ID de l'article
            article_id = article.get('id', '').replace('post-', '')
            
            # Catégories CSS
            categories = []
            class_list = article.get('class', [])
            for cls in class_list:
                if cls.startswith('category-'):
                    cat_name = cls.replace('category-', '')
                    categories.append(cat_name)
            
            # Date de publication
            time_tag = article.find('time', class_='entry-date')
            published_date = time_tag.get('datetime') if time_tag else None
            
            # Extrait de contenu
            content_div = article.find('div', class_='entry-content')
            excerpt = None
            if content_div:
                p_tag = content_div.find('p')
                if p_tag:
                    excerpt = p_tag.get_text(strip=True)
            
            return {
                'id': article_id,
                'title': title,
                'url': url,
                'categories': categories,
                'published_date': published_date,
                'excerpt': excerpt,
                'scraped_from': 'author_page'
            }
            
        except Exception as e:
            print(f"   ⚠️ Erreur extraction: {e}")
            return None
    
    def compare_with_previous(self, author_articles: List[Dict], 
                             option3_file: str) -> Dict:
        """
        Comparer avec le corpus existant (Option 3)
        
        Returns:
            Dict avec statistiques de comparaison
        """
        print("\n" + "="*60)
        print("🔍 COMPARAISON AVEC CORPUS EXISTANT")
        print("="*60)
        
        # Charger Option 3
        option3_ids = set()
        if Path(option3_file).exists():
            with open(option3_file, 'r', encoding='utf-8') as f:
                for line in f:
                    article = json.loads(line)
                    option3_ids.add(article['id'])
        
        print(f"\n📊 Corpus existant (Option 3): {len(option3_ids)} articles")
        
        # Analyser page auteur
        author_ids = set(a['id'] for a in author_articles)
        print(f"📊 Page auteur: {len(author_ids)} articles")
        
        # Intersection et différences
        common = option3_ids & author_ids
        only_option3 = option3_ids - author_ids
        only_author = author_ids - option3_ids
        
        print(f"\n✅ Communs: {len(common)} articles")
        print(f"🆕 Nouveaux (page auteur): {len(only_author)} articles")
        print(f"⚠️ Manquants (page auteur): {len(only_option3)} articles")
        print(f"📦 Total potentiel: {len(option3_ids | author_ids)} articles uniques")
        
        # Analyser les nouveaux articles
        if only_author:
            print(f"\n🆕 NOUVEAUX ARTICLES À SCRAPER:")
            new_articles = [a for a in author_articles if a['id'] in only_author]
            
            # Compter par catégorie
            from collections import Counter
            new_categories = []
            for article in new_articles:
                new_categories.extend(article.get('categories', []))
            
            cat_counts = Counter(new_categories)
            print("\nRépartition des nouveaux articles:")
            for cat, count in cat_counts.most_common():
                print(f"  - {cat}: {count} articles")
        
        return {
            'option3_count': len(option3_ids),
            'author_count': len(author_ids),
            'common_count': len(common),
            'new_articles': len(only_author),
            'missing_articles': len(only_option3),
            'total_unique': len(option3_ids | author_ids),
            'new_article_list': [a for a in author_articles if a['id'] in only_author]
        }
    
    def save_results(self, articles: List[Dict], output_file: str):
        """Sauvegarder les résultats"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
        
        print(f"\n💾 Sauvegardé: {output_file}")


def main():
    """Script principal - Exploration page auteur"""
    print("="*60)
    print("🚀 EXPLORATION: Page Auteur")
    print("="*60 + "\n")
    
    # Initialiser le scraper
    scraper = WolofOnlineAuthorScraper()
    
    # Scraper la page auteur
    author_articles = scraper.scrape_author_page(author_id=1)
    
    # Sauvegarder les métadonnées
    output_file = "data/raw/author_page_articles.jsonl"
    scraper.save_results(author_articles, output_file)
    
    # Comparer avec Option 3
    option3_file = "data/processed/option3_full_corpus.jsonl"
    comparison = scraper.compare_with_previous(author_articles, option3_file)
    
    # Recommandation
    print("\n" + "="*60)
    print("💡 RECOMMANDATION")
    print("="*60)
    
    if comparison['new_articles'] > 0:
        print(f"\n✅ {comparison['new_articles']} nouveaux articles identifiés !")
        print(f"\n🎯 Prochaine étape:")
        print(f"   Scraper le contenu complet de ces {comparison['new_articles']} articles")
        print(f"   pour augmenter le volume du corpus.")
        
        # Sauvegarder la liste des nouveaux articles
        new_file = "data/raw/new_articles_to_scrape.jsonl"
        scraper.save_results(comparison['new_article_list'], new_file)
        print(f"\n💾 Liste des nouveaux articles: {new_file}")
    else:
        print("\n⚠️ Aucun nouvel article trouvé via page auteur.")
        print("   Le corpus existant est déjà complet pour cet auteur.")
    
    print("\n✅ Exploration terminée!")


if __name__ == "__main__":
    main()
