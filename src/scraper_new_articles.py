# src/scraper_new_articles.py

import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import json
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs


class NewArticlesScraper:
    """Scraper pour les nouveaux articles identifiés"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Academic Research Bot) Wolof Corpus Collection'
        }
    
    def load_new_articles(self, input_file: str) -> List[Dict]:
        """Charger la liste des nouveaux articles à scraper"""
        articles = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                articles.append(json.loads(line))
        return articles
    
    def scrape_article_content(self, article: Dict) -> Dict:
        """Scraper le contenu complet d'un article"""
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
    
    def scrape_all_new_articles(self, articles: List[Dict]) -> List[Dict]:
        """Scraper tous les nouveaux articles"""
        total = len(articles)
        enriched_articles = []
        
        print(f"🚀 Scraping de {total} nouveaux articles...")
        print("="*60 + "\n")
        
        for idx, article in enumerate(articles, 1):
            print(f"📄 [{idx}/{total}] {article.get('title', 'Sans titre')[:60]}...")
            
            # Scraper l'article
            enriched = self.scrape_article_content(article)
            enriched_articles.append(enriched)
            
            # Statut
            if enriched.get('scraping_status') == 'success':
                word_count = enriched.get('word_count', 0)
                print(f"   ✅ {word_count} mots")
            else:
                print(f"   ❌ Échec: {enriched.get('scraping_error', 'Erreur inconnue')}")
            
            # Respecter le serveur
            time.sleep(1.5)
        
        return enriched_articles
    
    def merge_with_existing_corpus(self, new_articles: List[Dict], 
                                   existing_file: str) -> List[Dict]:
        """Fusionner avec le corpus existant"""
        print("\n" + "="*60)
        print("🔗 FUSION AVEC CORPUS EXISTANT")
        print("="*60)
        
        # Charger corpus existant
        existing_articles = []
        if Path(existing_file).exists():
            with open(existing_file, 'r', encoding='utf-8') as f:
                for line in f:
                    existing_articles.append(json.loads(line))
        
        print(f"\n📊 Corpus existant: {len(existing_articles)} articles")
        print(f"📊 Nouveaux articles: {len(new_articles)} articles")
        
        # Créer un dictionnaire par ID pour éviter les doublons
        corpus_dict = {a['id']: a for a in existing_articles}
        
        # Ajouter les nouveaux articles
        added = 0
        for article in new_articles:
            if article['id'] not in corpus_dict:
                corpus_dict[article['id']] = article
                added += 1
        
        merged_corpus = list(corpus_dict.values())
        
        print(f"📦 Corpus fusionné: {len(merged_corpus)} articles uniques")
        print(f"🆕 Nouveaux ajoutés: {added} articles")
        
        return merged_corpus
    
    def save_corpus(self, articles: List[Dict], output_file: str):
        """Sauvegarder le corpus"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
        
        print(f"\n💾 Corpus sauvegardé: {output_file}")
    
    def generate_statistics(self, articles: List[Dict], label: str = ""):
        """Générer des statistiques sur le corpus"""
        print("\n" + "="*60)
        print(f"📊 STATISTIQUES {label}")
        print("="*60)
        
        # Articles avec succès
        successful = [a for a in articles if a.get('scraping_status') == 'success']
        failed = len(articles) - len(successful)
        
        print(f"\n📈 Articles:")
        print(f"  - Total: {len(articles)}")
        print(f"  - Scrapés: {len(successful)}")
        print(f"  - Échecs: {failed}")
        
        if successful:
            # Statistiques de contenu
            total_words = sum(a.get('word_count', 0) for a in successful)
            avg_words = total_words / len(successful)
            
            print(f"\n📝 Contenu texte:")
            print(f"  - Total mots: {total_words:,}")
            print(f"  - Moyenne par article: {avg_words:.0f} mots")
            
            # Estimation en MB (UTF-8)
            estimated_bytes = total_words * 7  # ~7 bytes/mot en wolof
            estimated_mb = estimated_bytes / (1024 * 1024)
            print(f"  - Volume estimé: {estimated_mb:.2f} MB")
            
            # Vidéos
            articles_with_video = [a for a in successful if a.get('has_video')]
            total_videos = sum(len(a.get('youtube_videos', [])) for a in successful)
            
            print(f"\n🎥 Vidéos YouTube:")
            print(f"  - Articles avec vidéo: {len(articles_with_video)}")
            print(f"  - Total vidéos: {total_videos}")
            
            # Par catégorie
            from collections import Counter
            all_categories = []
            for article in successful:
                all_categories.extend(article.get('categories', []))
            
            cat_counts = Counter(all_categories)
            if cat_counts:
                print(f"\n📂 Top catégories:")
                for cat, count in cat_counts.most_common(10):
                    print(f"  - {cat}: {count} articles")


def main():
    """Script principal - Scraper les nouveaux articles"""
    print("="*60)
    print("🚀 SCRAPING NOUVEAUX ARTICLES")
    print("="*60 + "\n")
    
    # Initialiser le scraper
    scraper = NewArticlesScraper()
    
    # Charger les nouveaux articles
    new_articles_file = "data/raw/new_articles_to_scrape.jsonl"
    new_articles = scraper.load_new_articles(new_articles_file)
    
    print(f"📂 {len(new_articles)} nouveaux articles à scraper\n")
    
    # Scraper le contenu complet
    enriched_articles = scraper.scrape_all_new_articles(new_articles)
    
    # Statistiques des nouveaux articles
    scraper.generate_statistics(enriched_articles, "NOUVEAUX ARTICLES")
    
    # Fusionner avec le corpus existant
    existing_file = "data/processed/option3_full_corpus.jsonl"
    merged_corpus = scraper.merge_with_existing_corpus(enriched_articles, existing_file)
    
    # Sauvegarder le corpus mis à jour
    output_file = "data/processed/corpus_final_v2.jsonl"
    scraper.save_corpus(merged_corpus, output_file)
    
    # Statistiques du corpus complet
    scraper.generate_statistics(merged_corpus, "CORPUS COMPLET V2")
    
    print("\n" + "="*60)
    print("✅ SCRAPING TERMINÉ")
    print("="*60)
    print(f"\n💡 Prochaines étapes:")
    print(f"  1. Comparer avec les 40 MB d'Alwaly")
    print(f"  2. Filtrage qualité (fastText ≥70% wolof)")
    print(f"  3. Déduplication finale")
    print(f"  4. Validation humaine (200 textes)")


if __name__ == "__main__":
    main()
