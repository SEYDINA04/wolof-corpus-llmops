# transcribe_with_whosper.py

from whosper import WhosperTranscriber
from pathlib import Path
import json


def transcribe_videos():
    """Transcrire vidéos avec Whosper (Wolof optimized)"""
    
    print("="*60)
    print("🎤 TRANSCRIPTION WOLOF AVEC WHOSPER-LARGE-V2")
    print("="*60 + "\n")
    
    # Initialiser le transcriber
    print("📥 Chargement modèle CAYTU/whosper-large-v2...")
    transcriber = WhosperTranscriber(model_id="CAYTU/whosper-large-v2")
    print("✅ Modèle chargé\n")
    
    video_dir = Path("data/youtube_videos")
    output_dir = Path("data/youtube_transcriptions_wolof")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Lister les vidéos
    videos = list(video_dir.glob("*.mp4"))
    
    if not videos:
        print("❌ Aucune vidéo trouvée")
        return
    
    print(f"🎥 Vidéos à transcrire: {len(videos)}\n")
    
    transcriptions = {}
    
    for idx, video_path in enumerate(videos, 1):
        video_id = video_path.stem
        
        print(f"{idx}/{len(videos)} 🎤 {video_path.name}")
        
        try:
            # Transcrire
            result = transcriber.transcribe_audio(str(video_path))
            
            print(f"   ✅ Succès!")
            print(f"   📝 Texte: {result['text'][:100]}...")
            
            # Sauvegarder
            transcriptions[video_id] = {
                'text': result['text'],
                'language': 'wo',
                'model': 'CAYTU/whosper-large-v2'
            }
            
            # Export individual SRT/JSON
            save_transcription(video_id, result, output_dir)
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
    
    # Export résumé
    export_summary(transcriptions, output_dir)
    
    print("\n" + "="*60)
    print("✅ TRANSCRIPTION TERMINÉE")
    print("="*60)


def save_transcription(video_id, result, output_dir):
    """Sauvegarder transcription en JSON"""
    
    json_file = output_dir / f"{video_id}.json"
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'video_id': video_id,
            'text': result.get('text'),
            'language': 'wo',
            'model': 'CAYTU/whosper-large-v2'
        }, f, indent=2, ensure_ascii=False)


def export_summary(transcriptions, output_dir):
    """Exporter résumé"""
    
    summary_file = output_dir / "transcriptions_summary.json"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(transcriptions, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Résumé: {summary_file}")


if __name__ == "__main__":
    transcribe_videos()
