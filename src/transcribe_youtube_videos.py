# transcribe_youtube_videos.py

"""
Transcrire les vidéos YouTube avec Whisper (OpenAI)
Supporte: Wolof, multilingual
Output: SRT + JSON + TXT
"""

import subprocess
import json
from pathlib import Path


def install_whisper():
    """Installer Whisper si nécessaire"""
    print("📦 Installation Whisper...\n")
    subprocess.run(["pip", "install", "openai-whisper"], check=False)


def transcribe_videos():
    """Transcrire toutes les vidéos avec Whisper"""
    
    print("="*60)
    print("🎤 TRANSCRIPTION VIDÉOS YOUTUBE AVEC WHISPER")
    print("="*60 + "\n")
    
    video_dir = Path("data/youtube_videos")
    output_dir = Path("data/youtube_transcriptions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Lister les vidéos
    videos = list(video_dir.glob("*.mp4"))
    
    if not videos:
        print("❌ Aucune vidéo trouvée dans data/youtube_videos/")
        return
    
    print(f"📂 Vidéos trouvées: {len(videos)}\n")
    
    # Transcrire chaque vidéo
    transcriptions = {}
    
    for idx, video_path in enumerate(videos, 1):
        video_id = video_path.stem
        print(f"\n{idx}/{len(videos)} 🎥 Transcription: {video_path.name}")
        
        try:
            # Lancer Whisper
            cmd = [
                "whisper",
                str(video_path),
                "--model", "base",  # ou "tiny", "small", "medium", "large"
                "--language", "wo",  # Wolof
                "--output_format", "all",  # srt, vtt, txt, json, tsv
                "--output_dir", str(output_dir),
                "--device", "cuda",  # GPU (si dispo) ou "cpu"
            ]
            
            print(f"   ⏳ Processing... (durée estimée: ~{get_duration(video_path)})\n")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"   ✅ Succès!")
                
                # Lire le transcription JSON
                json_file = output_dir / f"{video_id}.json"
                if json_file.exists():
                    with open(json_file, 'r', encoding='utf-8') as f:
                        trans_data = json.load(f)
                    
                    transcriptions[video_id] = {
                        'title': video_id,
                        'language': 'wo',
                        'segments': trans_data.get('segments', [])
                    }
                    
                    # Afficher un aperçu
                    print(f"   📝 Segments: {len(trans_data.get('segments', []))}")
            else:
                print(f"   ❌ Erreur: {result.stderr}")
        
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    # Export résumé
    export_transcriptions(transcriptions, output_dir)


def get_duration(video_path: Path) -> str:
    """Estimer durée transcription"""
    import os
    import time
    
    try:
        stat = os.stat(video_path)
        # Whisper: ~1min audio = ~30sec (GPU)
        size_mb = stat.st_size / (1024 * 1024)
        # Approximation grossière: 289MB ~85min
        duration_min = (size_mb / 289) * 85
        trans_time_min = duration_min / 2  # GPU
        return f"{int(trans_time_min)}m"
    except:
        return "~50m"


def export_transcriptions(transcriptions: dict, output_dir: Path):
    """Exporter les transcriptions en JSON"""
    
    output_file = output_dir / "transcriptions_summary.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transcriptions, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Résumé exporté: {output_file}")


def create_srt_index():
    """Créer un index des fichiers SRT générés"""
    
    print("\n" + "="*60)
    print("📝 FICHIERS SRT GÉNÉRÉS")
    print("="*60 + "\n")
    
    output_dir = Path("data/youtube_transcriptions")
    srt_files = list(output_dir.glob("*.srt"))
    
    print(f"Fichiers SRT: {len(srt_files)}\n")
    
    for srt_file in srt_files:
        size_kb = srt_file.stat().st_size / 1024
        print(f"  ✅ {srt_file.name} ({size_kb:.1f} KB)")
        
        # Afficher premiers 3 segments
        with open(srt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:9]  # 3 segments
        
        print(f"     Aperçu:\n")
        for line in lines:
            if line.strip():
                print(f"     {line.strip()}")
        print()


def main():
    """Script principal"""
    print("🚀 WHISPER TRANSCRIPTION SETUP\n")
    
    # Installer Whisper
    install_whisper()
    
    # Transcriber
    transcribe_videos()
    
    # Index
    create_srt_index()
    
    print("\n" + "="*60)
    print("✅ TRANSCRIPTION TERMINÉE")
    print("="*60)
    print("\n💡 Fichiers générés:")
    print("  - *.srt (sous-titres)")
    print("  - *.vtt (WebVTT)")
    print("  - *.txt (texte brut)")
    print("  - *.json (données complètes)")
    print("  - transcriptions_summary.json (résumé)")


if __name__ == "__main__":
    main()
