# src/scraper.py

import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import json
import time
from pathlib import Path


class WolofOnlineScraper:
    """Scraper pour wolof-online.com"""

    def __init__(self, base_url: str = "https://www.wolof-online.com"):
        self.base_url = base_url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Academic Research Bot) Wolof Corpus Collection'
        }

    def scrape_homepage_pages(self, max_pages: int = 8) -> List[Dict]:
        """
        Option 1: Scraper les 8 pages d'accueil

        Returns:
            List[Dict]: Liste des articles avec titre, URL, catégorie
        """
        all_articles = []

        for page_num in range(1, max_pages + 1):
            print(f"📄 Scraping page {page_num}/{max_pages}...")

            # Construire l'URL de la page
            if page_num == 1:
                url = self.base_url
            else:
                url = f"{self.base_url}/?paged={page_num}"

            try:
                # Requête HTTP
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()

                # Parser le HTML
                soup = BeautifulSoup(response.content, 'lxml')

                # Extraire tous les articles
                articles = soup.find_all('article', class_='posts-entry')

                for article in articles:
                    article_data = self._extract_article_metadata(article)
                    if article_data:
                        all_articles.append(article_data)

                print(f"   ✅ {len(articles)} articles trouvés sur page {page_num}")

                # Respecter le serveur (1 sec entre requêtes)
                time.sleep(1)

            except Exception as e:
                print(f"   ❌ Erreur sur page {page_num}: {e}")
                continue

        print(f"\n🎯 Total: {len(all_articles)} articles collectés")
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

            # ID de l'article (depuis l'attribut id="post-XXXX")
            article_id = article.get('id', '').replace('post-', '')

            # Catégories (depuis les classes CSS)
            categories = []
            class_list = article.get('class', [])
            for cls in class_list:
                if cls.startswith('category-'):
                    cat_name = cls.replace('category-', '')
                    categories.append(cat_name)

            # Date de publication
            time_tag = article.find('time', class_='entry-date')
            published_date = time_tag.get('datetime') if time_tag else None

            # Extrait de contenu (si disponible sur la page d'accueil)
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
                'scraped_from': 'homepage'
            }

        except Exception as e:
            print(f"   ⚠️ Erreur extraction métadonnées: {e}")
            return None

    def save_to_jsonl(self, articles: List[Dict], output_path: str):
        """Sauvegarder les articles en format JSONL"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')

        print(f"💾 Sauvegardé dans: {output_path}")


def main():
    """Script principal Option 1"""
    print("="*60)
    print("🚀 OPTION 1: Scraping Pages d'Accueil (1-8)")
    print("="*60 + "\n")

    # Initialiser le scraper
    scraper = WolofOnlineScraper()

    # Scraper les 8 pages
    articles = scraper.scrape_homepage_pages(max_pages=8)

    # Sauvegarder les résultats
    output_path = "data/raw/option1_homepage_articles.jsonl"
    scraper.save_to_jsonl(articles, output_path)

    # Statistiques
    print("\n" + "="*60)
    print("📊 STATISTIQUES")
    print("="*60)
    print(f"Total articles: {len(articles)}")

    # Compter par catégorie
    from collections import Counter
    all_categories = []
    for article in articles:
        all_categories.extend(article.get('categories', []))

    category_counts = Counter(all_categories)
    print("\nRépartition par catégorie:")
    for cat, count in category_counts.most_common():
        print(f"  - {cat}: {count} articles")

    print("\n✅ Option 1 terminée!")


if __name__ == "__main__":
    main()
