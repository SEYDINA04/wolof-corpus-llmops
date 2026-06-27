# analyze_youtube_content.py

"""
Analyse les 4 vidéos YouTube trouvées dans le corpus wolof-online.com
Extrait métadonnées, télécharge vidéos, et prépare pour transcription ASR
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def check_dependencies():
    """Vérifier les dépendances nécessaires"""
    print("🔍 Vérification des dépendances...\n")

    # yt-dlp
    try:
        import yt_dlp
        print("✅ yt-dlp installé")
    except ImportError:
        print("❌ yt-dlp manquant")
        print("   Install: pip install yt-dlp")
        return False

    return True


def get_video_metadata(video_id: str) -> dict:
    """Extraire métadonnées d'une vidéo YouTube"""
    print(f"\n📥 Extraction métadonnées: {video_id}")

    try:
        import yt_dlp

        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }

        url = f"https://www.youtube.com/watch?v={video_id}"

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        metadata = {
            'video_id': video_id,
            'title': info.get('title', 'N/A'),
            'duration_seconds': info.get('duration', 0),
            'duration_formatted': format_duration(info.get('duration', 0)),
            'uploader': info.get('uploader', 'Unknown'),
            'view_count': info.get('view_count', 0),
            'like_count': info.get('like_count', 0),
            'upload_date': info.get('upload_date', 'Unknown'),
            'description': info.get('description', '')[:200] + '...',
            'url': url,
            'available': True
        }

        print(f"   ✅ Titre: {metadata['title']}")
        print(f"   ⏱️  Durée: {metadata['duration_formatted']}")
        print(f"   👤 Uploader: {metadata['uploader']}")

        return metadata

    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return {
            'video_id': video_id,
            'available': False,
            'error': str(e)
        }


def format_duration(seconds: int) -> str:
    """Formater la durée en HH:MM:SS"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def download_video(video_id: str, output_dir: str = "data/youtube_videos"):
    """Télécharger une vidéo YouTube"""
    print(f"\n📥 Téléchargement vidéo: {video_id}")

    try:
        import yt_dlp

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': str(output_path / f'{video_id}.%(ext)s'),
            'quiet': False,
            'progress_hooks': [progress_hook],
        }

        url = f"https://www.youtube.com/watch?v={video_id}"

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        downloaded_file = output_path / f"{video_id}.mp4"
        if downloaded_file.exists():
            size_mb = downloaded_file.stat().st_size / (1024 * 1024)
            print(f"   ✅ Téléchargé: {size_mb:.1f} MB")
            return str(downloaded_file)

    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return None


def progress_hook(d):
    """Hook pour afficher la progression du téléchargement"""
    if d['status'] == 'downloading':
        percent = d.get('_percent_str', 'N/A')
        speed = d.get('_speed_str', 'N/A')
        eta = d.get('_eta_str', 'N/A')
        print(f"\r   ⏳ {percent} à {speed} (ETA: {eta})", end='', flush=True)
    elif d['status'] == 'finished':
        print(f"\n   ✅ Téléchargement terminé")


def generate_transcription_plan(videos_metadata: list):
    """Générer un plan de transcription ASR"""
    print("\n" + "="*60)
    print("📋 PLAN DE TRANSCRIPTION ASR")
    print("="*60 + "\n")

    total_duration = 0

    for idx, meta in enumerate(videos_metadata, 1):
        if meta['available']:
            print(f"\n{idx}. {meta['title']}")
            print(f"   Video ID: {meta['video_id']}")
            print(f"   Durée: {meta['duration_formatted']}")
            print(f"   URL: {meta['url']}")
            print(f"   Uploader: {meta['uploader']}")
            print(f"   Apparaît dans: {get_article_count(meta['video_id'])} articles")

            total_duration += meta['duration_seconds']

    print(f"\n" + "="*60)
    print(f"⏱️  DURÉE TOTALE: {format_duration(total_duration)}")
    print("="*60)

    # Estimation temps transcription
    # Whisper: ~1min audio = ~30sec transcription (sur GPU)
    estimated_time = total_duration / 2  # En secondes
    print(f"\n⚡ Temps transcription estimé (GPU): ~{format_duration(int(estimated_time))}")
    print(f"⚡ Temps transcription estimé (CPU): ~{format_duration(int(estimated_time * 5))}")

    return total_duration


def get_article_count(video_id: str) -> int:
    """Compter combien d'articles contiennent cette vidéo"""
    try:
        import csv
        count = 0
        with open("youtube_videos_complete_list.csv", 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Video_ID'] == video_id:
                    count += 1
        return count
    except:
        return 0


def export_metadata(videos_metadata: list):
    """Exporter les métadonnées en JSON"""
    output_file = "data/youtube_videos_metadata.json"

    Path("data").mkdir(exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(videos_metadata, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Métadonnées exportées: {output_file}")


def main():
    """Script principal"""
    print("="*60)
    print("🎥 ANALYSE VIDÉOS YOUTUBE WOLOF-ONLINE")
    print("="*60 + "\n")

    # Vérifier dépendances
    if not check_dependencies():
        print("\n❌ Dépendances manquantes. Install yt-dlp:")
        print("   pip install yt-dlp")
        return

    # Liste des vidéos uniques
    video_ids = [
        "_eqpwNEXEzs",  # 65 occurrences
        "vrZIxImDf8Y",  # 64 occurrences
        "Z__rt9y5shc",  # 1 occurrence
        "P2UbcxLVhLw",  # 1 occurrence
    ]

    print(f"📊 Analyse de {len(video_ids)} vidéos uniques\n")

    # Extraire métadonnées
    videos_metadata = []
    for video_id in video_ids:
        meta = get_video_metadata(video_id)
        videos_metadata.append(meta)

    # Plan de transcription
    total_duration = generate_transcription_plan(videos_metadata)

    # Export métadonnées
    export_metadata(videos_metadata)

    # Proposer téléchargement
    print("\n" + "="*60)
    print("💾 TÉLÉCHARGER LES VIDÉOS ?")
    print("="*60)

    download_choice = input("\nTélécharger les vidéos pour transcription ? (y/n): ").lower()

    if download_choice == 'y':
        print("\n⚠️  Cela peut prendre du temps selon ta connexion...")

        for meta in videos_metadata:
            if meta['available']:
                download_video(meta['video_id'])

    print("\n" + "="*60)
    print("✅ ANALYSE TERMINÉE")
    print("="*60)
    print("\n💡 Prochaine étape:")
    print("   Transcrire les vidéos avec Whisper:")
    print("   pip install openai-whisper")
    print("   whisper data/youtube_videos/*.mp4 --language wo --output_format srt")


if __name__ == "__main__":
    main()
