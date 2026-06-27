#!/usr/bin/env python3
"""
Transcription Wolof avec Whosper-large-v2 + Disk Offloading
Optimisé pour machines avec RAM limitée
"""

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
from pathlib import Path
import json
import tempfile
import shutil
import torchaudio
import torch
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


def transcribe_videos_with_offload():
    """Transcrire vidéos avec offloading sur disque"""

    print("="*70)
    print("🎤 TRANSCRIPTION WOLOF AVEC WHOSPER-LARGE-V2 + DISK OFFLOADING")
    print("="*70 + "\n")

    # Créer un répertoire temporaire pour offloading
    offload_dir = tempfile.mkdtemp(prefix="whosper_offload_")
    print(f"💾 Offload dir: {offload_dir}\n")

    try:
        # ========== CHARGEMENT MODÈLE ==========
        print("📥 Chargement modèle CAYTU/whosper-large-v2...")

        # Charger le modèle base avec offloading
        base_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v2",
            device_map="auto",
            offload_folder=offload_dir,
            offload_state_dict=True,
            load_in_8bit=False,
            torch_dtype=torch.float16
        )

        # Charger l'adapter Whosper
        model = PeftModel.from_pretrained(base_model, "CAYTU/whosper-large-v2")

        # Charger le processor (pour l'audio)
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")

        print("✅ Modèle chargé avec succès\n")

        # ========== CONFIGURATION INFÉRENCE ==========
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        print(f"🖥️  Device: {device}")
        print(f"🔹 Modèle: Whosper-large-v2 (PEFT adapter)")
        print(f"🔹 Processor: Whisper (Mel-spectrogram)\n")

        # ========== PRÉPARATION FICHIERS ==========
        video_dir = Path("./data/youtube_videos")
        output_dir = Path("./data/youtube_transcriptions_wolof")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Lister les vidéos
        videos = sorted(list(video_dir.glob("*.mp4")))

        if not videos:
            print(f"❌ Aucune vidéo trouvée dans {video_dir}")
            return

        print(f"🎥 Vidéos trouvées: {len(videos)}\n")

        transcriptions = {}

        # ========== TRANSCRIPTION ==========
        for idx, video_path in enumerate(videos, 1):
            video_id = video_path.stem

            print(f"─" * 70)
            print(f"{idx}/{len(videos)} 🎤 Transcription: {video_path.name}")
            print(f"─" * 70)

            try:
                # Charger l'audio
                print(f"   📊 Chargement audio...")
                waveform, sr = torchaudio.load(str(video_path))

                # Rééchantillonner à 16kHz si nécessaire
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    waveform = resampler(waveform)

                # Mono
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                print(f"   ✅ Audio chargé ({waveform.shape})")

                # Préparer l'input pour le processor
                print(f"   🔄 Préparation features audio...")
                inputs = processor(
                    waveform[0].numpy(),
                    sampling_rate=16000,
                    return_tensors="pt"
                )

                input_features = inputs.input_features.to(torch.float16)
                print(f"   ✅ Features préparées")

                # Inférence
                print(f"   🧠 Inférence en cours...")
                with torch.no_grad():
                    predicted_ids = model.generate(
                        inputs=input_features,
                       # language="wo",
                        task="transcribe",
                        max_new_tokens=300,
                        num_beams=1,
                        do_sample=False
                    )

                # Décoder
                transcription = processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True
                )[0]

                print(f"   ✅ Inférence terminée")
                print(f"   📝 Texte: {transcription[:100]}...")

                # Sauvegarder
                transcriptions[video_id] = {
                    'text': transcription,
                    'language': 'wo',
                    'model': 'CAYTU/whosper-large-v2',
                    'device': device
                }

                # Export individuel (JSON)
                json_file = output_dir / f"{video_id}.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'video_id': video_id,
                        'text': transcription,
                        'language': 'wo',
                        'model': 'CAYTU/whosper-large-v2'
                    }, f, indent=2, ensure_ascii=False)

                print(f"   💾 Sauvegardé: {json_file}")
                print()

            except Exception as e:
                print(f"   ❌ Erreur: {e}\n")
                continue

        # ========== RÉSUMÉ ==========
        summary_file = output_dir / "transcriptions_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(transcriptions, f, indent=2, ensure_ascii=False)

        print("\n" + "="*70)
        print("✅ TRANSCRIPTION TERMINÉE")
        print("="*70)
        print(f"📊 Résumé: {summary_file}")
        print(f"📂 Output dir: {output_dir}")
        print(f"📈 Vidéos traitées: {len(transcriptions)}/{len(videos)}")
        print()

    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print(f"\n🧹 Nettoyage offload dir...")
        shutil.rmtree(offload_dir, ignore_errors=True)
        print("✅ Nettoyage terminé")


if __name__ == "__main__":
    transcribe_videos_with_offload()
