# src/scraper_full_content.py

import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Set
import json
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs


class WolofOnlineContentScraper:
    """Scraper du contenu complet des articles"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Academic Research Bot) Wolof Corpus Collection'
        }
    
    def load_unique_articles(self, option1_file: str, option2_file: str) -> List[Dict]:
        """
        Charger tous les articles uniques depuis Option 1 et Option 2
        
        Returns:
            List[Dict]: Liste des articles uniques avec leurs métadonnées
        """
        articles_dict = {}  # Utiliser dict pour dédupliquer par ID
        
        # Charger Option 1
        print("📂 Chargement Option 1...")
        if Path(option1_file).exists():
            with open(option1_file, 'r', encoding='utf-8') as f:
                for line in f:
                    article = json.loads(line)
                    articles_dict[article['id']] = article
        
        # Charger Option 2
        print("📂 Chargement Option 2...")
        if Path(option2_file).exists():
            with open(option2_file, 'r', encoding='utf-8') as f:
                for line in f:
                    article = json.loads(line)
                    # Si l'article existe déjà, enrichir avec info de catégorie
                    if article['id'] in articles_dict:
                        if 'primary_category' in article:
                            articles_dict[article['id']]['primary_category'] = article['primary_category']
                            articles_dict[article['id']]['domain'] = article['domain']
                    else:
                        articles_dict[article['id']] = article
        
        unique_articles = list(articles_dict.values())
        print(f"✅ {len(unique_articles)} articles uniques chargés\n")
        
        return unique_articles
    
    def scrape_article_content(self, article: Dict) -> Dict:
        """
        Scraper le contenu complet d'un article
        
        Args:
            article: Métadonnées de l'article (avec URL)
        
        Returns:
            Dict: Article enrichi avec contenu complet et vidéos
        """
        url = article.get('url')
        if not url:
            return article
        
        try:
            # Requête HTTP
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Parser le HTML
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Extraire le contenu principal
            content_data = self._extract_content(soup)
            
            # Extraire les vidéos YouTube
            videos = self._extract_youtube_videos(soup)
            
            # Enrichir l'article
            article['full_content'] = content_data['text']
            article['content_paragraphs'] = content_data['paragraphs']
            article['word_count'] = content_data['word_count']
            article['youtube_videos'] = videos
            article['has_video'] = len(videos) > 0
            article['scraping_status'] = 'success'
            
        except Exception as e:
            article['scraping_status'] = 'failed'
            article['scraping_error'] = str(e)
            article['full_content'] = None
            article['youtube_videos'] = []
        
        return article
    
    def _extract_content(self, soup: BeautifulSoup) -> Dict:
        """Extraire le contenu texte de l'article"""
        content_data = {
            'text': '',
            'paragraphs': [],
            'word_count': 0
        }
        
        # Chercher la div de contenu principal
        content_div = soup.find('div', class_='entry-content')
        
        if not content_div:
            return content_data
        
        # Extraire tous les paragraphes
        paragraphs = []
        for p in content_div.find_all('p'):
            text = p.get_text(strip=True)
            if text and len(text) > 10:  # Ignorer les très courts
                paragraphs.append(text)
        
        # Assembler le texte complet
        full_text = '\n\n'.join(paragraphs)
        word_count = len(full_text.split())
        
        content_data['text'] = full_text
        content_data['paragraphs'] = paragraphs
        content_data['word_count'] = word_count
        
        return content_data
    
    def _extract_youtube_videos(self, soup: BeautifulSoup) -> List[Dict]:
        """Extraire les vidéos YouTube de la page"""
        videos = []
        
        # Chercher tous les iframes YouTube
        iframes = soup.find_all('iframe')
        
        for iframe in iframes:
            src = iframe.get('src', '')
            
            # Vérifier si c'est une vidéo YouTube
            if 'youtube.com' in src or 'youtu.be' in src:
                video_id = self._extract_youtube_id(src)
                if video_id:
                    videos.append({
                        'video_id': video_id,
                        'embed_url': src,
                        'watch_url': f"https://www.youtube.com/watch?v={video_id}"
                    })
        
        return videos
    
    def _extract_youtube_id(self, url: str) -> str:
        """Extraire l'ID YouTube depuis une URL"""
        try:
            parsed = urlparse(url)
            
            # Format: youtube.com/embed/VIDEO_ID
            if 'youtube.com/embed/' in url:
                return url.split('/embed/')[-1].split('?')[0]
            
            # Format: youtube.com/watch?v=VIDEO_ID
            if 'youtube.com/watch' in url:
                query = parse_qs(parsed.query)
                return query.get('v', [None])[0]
            
            # Format: youtu.be/VIDEO_ID
            if 'youtu.be/' in url:
                return parsed.path.split('/')[-1]
            
        except Exception:
            pass
        
        return None
    
    def scrape_all_articles(self, articles: List[Dict], batch_size: int = 10) -> List[Dict]:
        """
        Scraper tous les articles avec gestion des erreurs
        
        Args:
            articles: Liste des articles à scraper
            batch_size: Afficher progression tous les X articles
        
        Returns:
            List[Dict]: Articles enrichis avec contenu complet
        """
        total = len(articles)
        enriched_articles = []
        
        print(f"🚀 Scraping du contenu de {total} articles...")
        print("="*60 + "\n")
        
        for idx, article in enumerate(articles, 1):
            # Scraper l'article
            enriched = self.scrape_article_content(article)
            enriched_articles.append(enriched)
            
            # Afficher progression
            if idx % batch_size == 0 or idx == total:
                success = sum(1 for a in enriched_articles if a.get('scraping_status') == 'success')
                failed = idx - success
                print(f"📊 Progression: {idx}/{total} articles | ✅ {success} réussis | ❌ {failed} échoués")
            
            # Respecter le serveur (1.5 sec entre requêtes)
            time.sleep(1.5)
        
        return enriched_articles
    
    def save_corpus(self, articles: List[Dict], output_file: str):
        """Sauvegarder le corpus final"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
        
        print(f"\n💾 Corpus sauvegardé: {output_file}")
    
    def generate_statistics(self, articles: List[Dict]):
        """Générer des statistiques sur le corpus"""
        print("\n" + "="*60)
        print("📊 STATISTIQUES CORPUS FINAL")
        print("="*60)
        
        # Statistiques générales
        total = len(articles)
        success = sum(1 for a in articles if a.get('scraping_status') == 'success')
        failed = total - success
        
        print(f"\n📈 Articles:")
        print(f"  - Total: {total}")
        print(f"  - Scrapés avec succès: {success}")
        print(f"  - Échecs: {failed}")
        
        # Statistiques de contenu
        successful_articles = [a for a in articles if a.get('scraping_status') == 'success']
        
        if successful_articles:
            total_words = sum(a.get('word_count', 0) for a in successful_articles)
            avg_words = total_words / len(successful_articles)
            
            print(f"\n📝 Contenu texte:")
            print(f"  - Total mots: {total_words:,}")
            print(f"  - Moyenne par article: {avg_words:.0f} mots")
            
            # Estimation en MB (UTF-8)
            # Approximation: 1 mot wolof ≈ 7 bytes en moyenne
            estimated_bytes = total_words * 7
            estimated_mb = estimated_bytes / (1024 * 1024)
            print(f"  - Volume estimé: {estimated_mb:.2f} MB")
        
        # Statistiques vidéos
        articles_with_video = [a for a in articles if a.get('has_video')]
        total_videos = sum(len(a.get('youtube_videos', [])) for a in articles)
        
        print(f"\n🎥 Vidéos YouTube:")
        print(f"  - Articles avec vidéo: {len(articles_with_video)}")
        print(f"  - Total vidéos: {total_videos}")
        
        # Statistiques par domaine
        domain_stats = {}
        for article in successful_articles:
            domain = article.get('domain', 'non_classifie')
            if domain not in domain_stats:
                domain_stats[domain] = {'count': 0, 'words': 0}
            domain_stats[domain]['count'] += 1
            domain_stats[domain]['words'] += article.get('word_count', 0)
        
        if domain_stats:
            print(f"\n📂 Par domaine:")
            for domain, stats in sorted(domain_stats.items(), key=lambda x: x[1]['words'], reverse=True):
                print(f"  - {domain}: {stats['count']} articles, {stats['words']:,} mots")


def main():
    """Script principal Option 3"""
    print("="*60)
    print("🚀 OPTION 3: Scraping Contenu Complet")
    print("="*60 + "\n")
    
    # Initialiser le scraper
    scraper = WolofOnlineContentScraper()
    
    # Charger les articles uniques
    option1_file = "data/raw/option1_homepage_articles.jsonl"
    option2_file = "data/raw/option2_all_categories.jsonl"
    
    unique_articles = scraper.load_unique_articles(option1_file, option2_file)
    
    # Scraper le contenu complet
    enriched_articles = scraper.scrape_all_articles(unique_articles, batch_size=10)
    
    # Sauvegarder le corpus final
    output_file = "data/processed/option3_full_corpus.jsonl"
    scraper.save_corpus(enriched_articles, output_file)
    
    # Générer les statistiques
    scraper.generate_statistics(enriched_articles)
    
    print("\n✅ Option 3 terminée!")
    print("\n💡 Prochaines étapes:")
    print("  1. Validation qualité (filtrage N1: fastText)")
    print("  2. Déduplication avec corpus Alwaly (40 MB)")
    print("  3. Validation humaine (échantillon 200 textes)")


if __name__ == "__main__":
    main()
