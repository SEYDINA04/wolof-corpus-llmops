# src/consolidate_wolof_online.py

import json
from pathlib import Path
from typing import List, Dict
from collections import Counter


class WolofOnlineConsolidator:
    """Consolidation du corpus wolof-online.com"""
    
    def filter_text_only(self, corpus_file: str) -> tuple[List[Dict], List[Dict]]:
        """
        Séparer articles avec texte et vidéos uniquement
        
        Returns:
            (articles_with_text, articles_video_only)
        """
        articles_with_text = []
        articles_video_only = []
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                article = json.loads(line)
                
                # Vérifier si l'article a du contenu textuel
                word_count = article.get('word_count', 0)
                
                if word_count > 0:
                    articles_with_text.append(article)
                else:
                    articles_video_only.append(article)
        
        return articles_with_text, articles_video_only
    
    def calculate_statistics(self, articles: List[Dict], label: str = ""):
        """Calculer et afficher les statistiques"""
        print("\n" + "="*60)
        print(f"📊 STATISTIQUES {label}")
        print("="*60)
        
        # Statistiques globales
        total_articles = len(articles)
        total_words = sum(a.get('word_count', 0) for a in articles)
        avg_words = total_words / total_articles if total_articles > 0 else 0
        
        print(f"\n📈 Volume:")
        print(f"  - Total articles: {total_articles:,}")
        print(f"  - Total mots: {total_words:,}")
        print(f"  - Moyenne par article: {avg_words:.0f} mots")
        
        # Estimation en MB (UTF-8)
        estimated_bytes = total_words * 7  # ~7 bytes/mot en wolof
        estimated_mb = estimated_bytes / (1024 * 1024)
        print(f"  - Volume estimé: {estimated_mb:.2f} MB")
        
        # Vidéos YouTube
        articles_with_video = [a for a in articles if a.get('has_video')]
        total_videos = sum(len(a.get('youtube_videos', [])) for a in articles)
        
        print(f"\n🎥 Vidéos YouTube:")
        print(f"  - Articles avec vidéo: {len(articles_with_video)}")
        print(f"  - Total vidéos: {total_videos}")
        
        # Par catégorie
        all_categories = []
        for article in articles:
            cats = article.get('categories', []) or []
            all_categories.extend(cats)
        
        if all_categories:
            cat_counts = Counter(all_categories)
            print(f"\n📂 Top 10 catégories:")
            for cat, count in cat_counts.most_common(10):
                cat_words = sum(a.get('word_count', 0) for a in articles 
                               if cat in a.get('categories', []))
                print(f"  - {cat}: {count} articles ({cat_words:,} mots)")
        
        # Distribution temporelle
        years = []
        for article in articles:
            date = article.get('published_date', '')
            if date:
                year = date.split('-')[0]
                years.append(year)
        
        if years:
            year_counts = Counter(years)
            print(f"\n📅 Distribution temporelle:")
            for year, count in sorted(year_counts.items(), reverse=True)[:5]:
                print(f"  - {year}: {count} articles")
        
        return {
            'total_articles': total_articles,
            'total_words': total_words,
            'total_mb': estimated_mb,
            'total_videos': total_videos
        }
    
    def save_corpus(self, articles: List[Dict], output_file: str):
        """Sauvegarder le corpus"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
        
        print(f"\n💾 Sauvegardé: {output_file}")
    
    def save_video_list(self, articles: List[Dict], output_file: str):
        """Sauvegarder la liste des vidéos pour exploration future"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        video_list = []
        for article in articles:
            videos = article.get('youtube_videos', [])
            for video in videos:
                video_list.append({
                    'article_id': article.get('id'),
                    'article_title': article.get('title'),
                    'video_id': video.get('video_id'),
                    'watch_url': video.get('watch_url'),
                    'categories': article.get('categories', [])
                })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for video in video_list:
                f.write(json.dumps(video, ensure_ascii=False) + '\n')
        
        print(f"💾 Liste vidéos: {output_file} ({len(video_list)} vidéos)")


def main():
    """Script principal - Consolidation wolof-online.com"""
    print("="*60)
    print("🚀 CONSOLIDATION CORPUS WOLOF-ONLINE.COM")
    print("="*60 + "\n")
    
    consolidator = WolofOnlineConsolidator()
    
    # Charger le corpus complet
    input_file = "data/processed/corpus_final_v2.jsonl"
    print(f"📂 Chargement: {input_file}\n")
    
    # Séparer texte et vidéo
    articles_text, articles_video = consolidator.filter_text_only(input_file)
    
    print("📊 Séparation texte / vidéo:")
    print(f"  - Articles avec texte: {len(articles_text)}")
    print(f"  - Articles vidéo uniquement: {len(articles_video)}")
    
    # Statistiques articles texte
    stats_text = consolidator.calculate_statistics(articles_text, "CORPUS TEXTUEL")
    
    # Statistiques articles vidéo
    stats_video = consolidator.calculate_statistics(articles_video, "ARTICLES VIDÉO (exclus)")
    
    # Sauvegarder corpus textuel final
    text_output = "data/final/wolof_online_text_corpus.jsonl"
    consolidator.save_corpus(articles_text, text_output)
    
    # Sauvegarder articles vidéo (pour référence)
    video_output = "data/final/wolof_online_video_articles.jsonl"
    consolidator.save_corpus(articles_video, video_output)
    
    # Sauvegarder liste des vidéos YouTube
    video_list_output = "data/final/youtube_videos_list.jsonl"
    all_articles = articles_text + articles_video
    consolidator.save_video_list(all_articles, video_list_output)
    
    # Résumé final
    print("\n" + "="*60)
    print("✅ CONSOLIDATION TERMINÉE")
    print("="*60)
    
    print(f"\n📦 FICHIERS GÉNÉRÉS:")
    print(f"  1. {text_output}")
    print(f"     → {stats_text['total_articles']} articles, {stats_text['total_mb']:.2f} MB")
    print(f"  2. {video_output}")
    print(f"     → {stats_video['total_articles']} articles vidéo (pour Phase 3)")
    print(f"  3. {video_list_output}")
    print(f"     → {stats_text['total_videos'] + stats_video['total_videos']} vidéos YouTube")
    
    print(f"\n🎯 CORPUS FINAL WOLOF-ONLINE:")
    print(f"  - Articles texte: {stats_text['total_articles']:,}")
    print(f"  - Total mots: {stats_text['total_words']:,}")
    print(f"  - Volume: {stats_text['total_mb']:.2f} MB")
    
    print(f"\n💡 PROCHAINES ÉTAPES:")
    print(f"  1. Comparer avec les 40 MB d'Alwaly (déduplication)")
    print(f"  2. Filtrage qualité (fastText ≥70% wolof)")
    print(f"  3. Validation humaine (200 textes)")
    print(f"  4. Livraison à l'équipe ML")
    
    print(f"\n📌 NOTE:")
    print(f"  Les {stats_video['total_videos']} vidéos YouTube sont sauvegardées")
    print(f"  pour exploration future (Phase 3 - transcription ASR)")


if __name__ == "__main__":
    main()
