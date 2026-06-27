# src/scraper_by_category.py

import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Set
import json
import time
from pathlib import Path
from collections import Counter


class WolofOnlineCategoryScraper:
    """Scraper par catégorie pour wolof-online.com"""

    def __init__(self, base_url: str = "https://www.wolof-online.com"):
        self.base_url = base_url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Academic Research Bot) Wolof Corpus Collection'
        }

        # Catégories identifiées depuis le menu
        self.categories = {
            'Politig': {'cat_id': 4, 'domain': 'politique'},
            'NEKKIN': {'cat_id': 12, 'domain': 'culture'},
            'Diine': {'cat_id': 15, 'domain': 'religion'},
            'Caada': {'cat_id': 6, 'domain': 'histoire'},
            'Tàggat yaram': {'cat_id': 14, 'domain': 'sante'}
        }

    def scrape_category(self, cat_name: str, cat_id: int, domain: str) -> List[Dict]:
        """
        Scraper une catégorie complète

        Args:
            cat_name: Nom de la catégorie (ex: 'Politig')
            cat_id: ID de la catégorie (ex: 4)
            domain: Domaine thématique (ex: 'politique')

        Returns:
            List[Dict]: Liste des articles de cette catégorie
        """
        all_articles = []
        page_num = 1

        print(f"\n📂 Catégorie: {cat_name} (Domaine: {domain})")
        print("-" * 60)

        while True:
            # Construire l'URL de la catégorie
            if page_num == 1:
                url = f"{self.base_url}/?cat={cat_id}"
            else:
                url = f"{self.base_url}/?cat={cat_id}&paged={page_num}"

            try:
                # Requête HTTP
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()

                # Parser le HTML
                soup = BeautifulSoup(response.content, 'lxml')

                # Extraire les articles
                articles = soup.find_all('article', class_='posts-entry')

                # Si aucun article, on a atteint la fin
                if not articles:
                    break

                print(f"   Page {page_num}: {len(articles)} articles")

                # Extraire métadonnées
                for article in articles:
                    article_data = self._extract_article_metadata(article, cat_name, domain)
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

        print(f"   ✅ Total: {len(all_articles)} articles dans {cat_name}")
        return all_articles

    def _extract_article_metadata(self, article, category_name: str, domain: str) -> Dict:
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

            # Catégories depuis les classes CSS
            css_categories = []
            class_list = article.get('class', [])
            for cls in class_list:
                if cls.startswith('category-'):
                    cat_name = cls.replace('category-', '')
                    css_categories.append(cat_name)

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
                'primary_category': category_name,
                'domain': domain,
                'all_categories': css_categories,
                'published_date': published_date,
                'excerpt': excerpt,
                'scraped_from': f'category_{category_name}'
            }

        except Exception as e:
            print(f"   ⚠️ Erreur extraction: {e}")
            return None

    def scrape_all_categories(self) -> Dict[str, List[Dict]]:
        """Scraper toutes les catégories"""
        results = {}

        for cat_name, cat_info in self.categories.items():
            articles = self.scrape_category(
                cat_name=cat_name,
                cat_id=cat_info['cat_id'],
                domain=cat_info['domain']
            )
            results[cat_name] = articles

        return results

    def compare_with_option1(self, option2_articles: Dict[str, List[Dict]],
                            option1_file: str) -> Dict:
        """
        Comparer avec les résultats de l'Option 1

        Returns:
            Dict avec statistiques de comparaison
        """
        print("\n" + "="*60)
        print("🔍 COMPARAISON OPTION 1 vs OPTION 2")
        print("="*60)

        # Charger Option 1
        option1_ids = set()
        with open(option1_file, 'r', encoding='utf-8') as f:
            for line in f:
                article = json.loads(line)
                option1_ids.add(article['id'])

        print(f"\n📊 Option 1: {len(option1_ids)} articles uniques")

        # Analyser Option 2
        option2_all = []
        option2_ids = set()

        for cat_name, articles in option2_articles.items():
            for article in articles:
                option2_all.append(article)
                option2_ids.add(article['id'])

        print(f"📊 Option 2: {len(option2_ids)} articles uniques")

        # Intersection et différences
        common = option1_ids & option2_ids
        only_option1 = option1_ids - option2_ids
        only_option2 = option2_ids - option1_ids

        print(f"\n✅ Communs: {len(common)} articles")
        print(f"🆕 Uniquement Option 1: {len(only_option1)} articles")
        print(f"🆕 Uniquement Option 2: {len(only_option2)} articles")
        print(f"📦 Total combiné: {len(option1_ids | option2_ids)} articles uniques")

        return {
            'option1_count': len(option1_ids),
            'option2_count': len(option2_ids),
            'common_count': len(common),
            'only_option1': len(only_option1),
            'only_option2': len(only_option2),
            'total_unique': len(option1_ids | option2_ids)
        }

    def save_results(self, category_results: Dict[str, List[Dict]], output_dir: str):
        """Sauvegarder les résultats par catégorie"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Sauvegarder chaque catégorie séparément
        for cat_name, articles in category_results.items():
            cat_file = output_path / f"category_{cat_name.lower().replace(' ', '_')}.jsonl"
            with open(cat_file, 'w', encoding='utf-8') as f:
                for article in articles:
                    f.write(json.dumps(article, ensure_ascii=False) + '\n')
            print(f"💾 {cat_name}: {len(articles)} articles → {cat_file.name}")

        # Sauvegarder tout dans un seul fichier
        all_file = output_path / "option2_all_categories.jsonl"
        with open(all_file, 'w', encoding='utf-8') as f:
            for articles in category_results.values():
                for article in articles:
                    f.write(json.dumps(article, ensure_ascii=False) + '\n')

        print(f"\n💾 Fichier consolidé: option2_all_categories.jsonl")


def main():
    """Script principal Option 2"""
    print("="*60)
    print("🚀 OPTION 2: Scraping par Catégorie")
    print("="*60)

    # Initialiser le scraper
    scraper = WolofOnlineCategoryScraper()

    # Scraper toutes les catégories
    category_results = scraper.scrape_all_categories()

    # Sauvegarder les résultats
    output_dir = "data/raw"
    scraper.save_results(category_results, output_dir)

    # Statistiques par domaine
    print("\n" + "="*60)
    print("📊 STATISTIQUES PAR DOMAINE")
    print("="*60)

    for cat_name, articles in category_results.items():
        domain = scraper.categories[cat_name]['domain']
        print(f"\n{cat_name} ({domain}):")
        print(f"  - {len(articles)} articles")

        # Distribution temporelle
        dates = [a['published_date'] for a in articles if a['published_date']]
        if dates:
            years = [d.split('-')[0] for d in dates]
            year_counts = Counter(years)
            print(f"  - Années: {dict(year_counts.most_common(3))}")

    # Comparer avec Option 1
    option1_file = "data/raw/option1_homepage_articles.jsonl"
    if Path(option1_file).exists():
        comparison = scraper.compare_with_option1(category_results, option1_file)

    print("\n✅ Option 2 terminée!")


if __name__ == "__main__":
    main()
