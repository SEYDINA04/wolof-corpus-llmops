# transcribe_with_whosper_offload.py

from whosper import WhosperTranscriber
from pathlib import Path
import json
import tempfile
import os


def transcribe_videos_with_offload():
    """Transcrire avec offloading sur disque"""
    
    print("="*60)
    print("🎤 TRANSCRIPTION WOLOF (AVEC DISK OFFLOADING)")
    print("="*60 + "\n")
    
    # Créer un répertoire temporaire pour offloading
    offload_dir = tempfile.mkdtemp(prefix="whosper_offload_")
    print(f"💾 Offload dir: {offload_dir}\n")
    
    try:
        print("📥 Chargement modèle CAYTU/whosper-large-v2...")
        
        # Solution : utiliser device_map avec offload_dir
        from transformers import WhisperForConditionalGeneration
        from peft import PeftModel
        
        # Charger le modèle base
        base_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v2",
            device_map="auto",
            offload_folder=offload_dir,  # ← KEY: Offload sur disque
            offload_state_dict=True
        )
        
        # Charger l'adapter Whosper
        model = PeftModel.from_pretrained(base_model, "CAYTU/whosper-large-v2")
        print("✅ Modèle chargé\n")
        
        # ... reste du script ...
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(offload_dir, ignore_errors=True)


if __name__ == "__main__":
    transcribe_videos_with_offload()
