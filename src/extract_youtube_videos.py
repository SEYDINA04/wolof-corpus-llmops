# extract_youtube_videos.py

import json
from pathlib import Path
from collections import defaultdict


def extract_youtube_videos():
    """Extraire et afficher toutes les vidéos YouTube trouvées"""
    
    print("="*60)
    print("🎥 VIDÉOS YOUTUBE TROUVÉES")
    print("="*60 + "\n")
    
    # Chemin du fichier
    video_file = "data/final/youtube_videos_list.jsonl"
    
    if not Path(video_file).exists():
        print(f"❌ Fichier introuvable: {video_file}")
        return
    
    # Charger les vidéos
    videos = []
    with open(video_file, 'r', encoding='utf-8') as f:
        for line in f:
            videos.append(json.loads(line))
    
    print(f"📊 Total vidéos trouvées: {len(videos):,}\n")
    
    # Grouper par article
    articles_with_videos = defaultdict(list)
    
    with open("data/validated/n1_wolof_passed.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            article = json.loads(line)
            article_id = article.get('id')
            videos_in_article = article.get('youtube_videos', [])
            
            if videos_in_article:
                articles_with_videos[article_id] = {
                    'title': article.get('title'),
                    'url': article.get('url'),
                    'videos': videos_in_article
                }
    
    # Afficher les vidéos avec leurs liens
    print("🎬 VIDÉOS PAR ARTICLE\n")
    
    count = 0
    for article_id, data in sorted(articles_with_videos.items()):
        print(f"\n📄 Article ID: {article_id}")
        print(f"   Titre: {data['title'][:80]}...")
        print(f"   Source: {data['url']}")
        print(f"   Vidéos: {len(data['videos'])}")
        
        for i, video in enumerate(data['videos'], 1):
            video_id = video.get('video_id')
            watch_url = video.get('watch_url')
            embed_url = video.get('embed_url')
            
            count += 1
            print(f"\n   {count}. 🎥 ID: {video_id}")
            print(f"      Watch: {watch_url}")
            print(f"      Embed: {embed_url}")
    
    # Résumé
    print("\n" + "="*60)
    print(f"✅ RÉSUMÉ")
    print("="*60)
    print(f"  - Total vidéos: {count}")
    print(f"  - Articles avec vidéos: {len(articles_with_videos)}")
    print(f"  - Moyenne vidéos/article: {count / len(articles_with_videos):.1f}")
    
    # Exporter dans un fichier CSV pour facilité
    export_to_csv(articles_with_videos, count)


def export_to_csv(articles_with_videos, total_videos):
    """Exporter les vidéos en CSV"""
    import csv
    
    csv_file = "data/final/youtube_videos_complete_list.csv"
    
    print(f"\n💾 Export CSV: {csv_file}")
    
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Video_ID', 'Watch_URL', 'Embed_URL', 'Article_ID', 'Article_Title', 'Article_Source'])
        
        count = 0
        for article_id, data in sorted(articles_with_videos.items()):
            for video in data['videos']:
                writer.writerow([
                    video.get('video_id'),
                    video.get('watch_url'),
                    video.get('embed_url'),
                    article_id,
                    data['title'],
                    data['url']
                ])
                count += 1
    
    print(f"✅ {count} vidéos exportées en CSV")


if __name__ == "__main__":
    extract_youtube_videos()
