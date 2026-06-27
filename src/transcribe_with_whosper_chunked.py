#!/usr/bin/env python3
"""
Transcription Wolof avec Whosper-large-v2 + CHUNKING (10-15s segments)
Optimisé pour vidéos longues (1h+)
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

warnings.filterwarnings('ignore')


def transcribe_long_audio_chunked(
    video_path,
    model,
    processor,
    device,
    chunk_duration=12,  # 10-15s (default 12s)
    stride=11  # Overlap de 1s pour continuité
):
    """
    Transcrire vidéo longue avec chunking

    Args:
        video_path: Chemin vers vidéo
        model: Modèle Whosper chargé
        processor: Whisper processor
        device: 'cpu' ou 'cuda'
        chunk_duration: Durée chunk en secondes (10-15)
        stride: Chevauchement entre chunks (secondes)

    Returns:
        str: Transcription complète
    """

    # Charger audio
    print(f"   📊 Chargement audio...")
    waveform, sr = torchaudio.load(str(video_path))

    # Rééchantillonner à 16kHz
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
        sr = 16000

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    total_duration = waveform.shape[1] / sr
    print(f"   ✅ Audio chargé ({total_duration:.1f}s)")

    # Calculer chunks
    chunk_samples = int(chunk_duration * sr)
    stride_samples = int(stride * sr)

    num_chunks = (waveform.shape[1] - chunk_samples) // stride_samples + 1
    print(f"   📈 Chunks: {num_chunks} x {chunk_duration}s")

    all_texts = []

    # Traiter chaque chunk
    for idx in range(num_chunks):
        start_sample = idx * stride_samples
        end_sample = start_sample + chunk_samples

        # Extraire chunk
        chunk = waveform[:, start_sample:end_sample]

        # Temps
        chunk_time = f"{start_sample / sr:.1f}s - {end_sample / sr:.1f}s"
        print(f"      [{idx + 1}/{num_chunks}] {chunk_time}", end=" ", flush=True)

        try:
            # Préparer features
            inputs = processor(
                chunk[0].numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            )

            input_features = inputs.input_features.to(device).to(torch.float16)

            # Inférence
            with torch.no_grad():
                predicted_ids = model.generate(
                    inputs=input_features,
                    task="transcribe",
                    max_new_tokens=300,
                    num_beams=1,
                    do_sample=False
                )

            # Décoder
            text = processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0].strip()

            if text:
                all_texts.append(text)
                print(f"✅ ({len(text)} chars)")
            else:
                print(f"⚠️  (vide)")

        except Exception as e:
            print(f"❌ {e}")
            continue

    # Concaténer avec espaces
    full_text = " ".join(all_texts)

    return full_text


def transcribe_videos_chunked():
    """Transcrire vidéos avec chunking"""

    print("="*70)
    print("🎤 TRANSCRIPTION WOLOF + CHUNKING 10-15s")
    print("="*70 + "\n")

    offload_dir = tempfile.mkdtemp(prefix="whosper_offload_")
    print(f"💾 Offload dir: {offload_dir}\n")

    try:
        # ========== CHARGEMENT MODÈLE ==========
        print("📥 Chargement modèle CAYTU/whosper-large-v2...")

        base_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v2",
            device_map="auto",
            offload_folder=offload_dir,
            offload_state_dict=True,
            load_in_8bit=False,
            torch_dtype=torch.float16
        )

        model = PeftModel.from_pretrained(base_model, "CAYTU/whosper-large-v2")
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")

        print("✅ Modèle chargé\n")

        # ========== CONFIGURATION ==========
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        print(f"🖥️  Device: {device}")
        print(f"🔹 Modèle: Whosper-large-v2")
        print(f"🔹 Chunking: 12s avec overlap 1s\n")

        # ========== FICHIERS ==========
        video_dir = Path("./data/youtube_videos")
        output_dir = Path("./data/youtube_transcriptions_wolof_chunked")
        output_dir.mkdir(parents=True, exist_ok=True)

        videos = sorted(list(video_dir.glob("*.mp4")))

        if not videos:
            print(f"❌ Aucune vidéo trouvée")
            return

        print(f"🎥 Vidéos: {len(videos)}\n")

        transcriptions = {}

        # ========== TRANSCRIPTION ==========
        for idx, video_path in enumerate(videos, 1):
            video_id = video_path.stem

            print(f"─" * 70)
            print(f"{idx}/{len(videos)} 🎤 {video_path.name}")
            print(f"─" * 70)

            try:
                # Transcrire avec chunking
                full_text = transcribe_long_audio_chunked(
                    video_path,
                    model,
                    processor,
                    device,
                    chunk_duration=12,  # 12s chunks
                    stride=11  # 1s overlap
                )

                print(f"\n   ✅ Transcription terminée")
                print(f"   📊 Texte final: {len(full_text)} caractères")
                print(f"   📝 Aperçu: {full_text[:150]}...\n")

                # Sauvegarder
                transcriptions[video_id] = {
                    'text': full_text,
                    'language': 'wo',
                    'model': 'CAYTU/whosper-large-v2',
                    'device': device,
                    'chunk_duration': 12,
                    'total_chars': len(full_text)
                }

                json_file = output_dir / f"{video_id}.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'video_id': video_id,
                        'text': full_text,
                        'language': 'wo',
                        'model': 'CAYTU/whosper-large-v2',
                        'chunk_duration': 12
                    }, f, indent=2, ensure_ascii=False)

                print(f"   💾 Sauvegardé: {json_file}\n")

            except Exception as e:
                print(f"   ❌ Erreur: {e}\n")
                continue

        # ========== RÉSUMÉ ==========
        summary_file = output_dir / "transcriptions_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(transcriptions, f, indent=2, ensure_ascii=False)

        print("\n" + "="*70)
        print("✅ TRANSCRIPTION CHUNKED TERMINÉE")
        print("="*70)
        print(f"📊 Résumé: {summary_file}")
        print(f"📂 Output: {output_dir}")
        print(f"📈 Vidéos: {len(transcriptions)}/{len(videos)}")

        # Stats
        for vid_id, data in transcriptions.items():
            print(f"   • {vid_id}: {data['total_chars']} chars")

        print()

    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print(f"\n🧹 Cleanup...")
        shutil.rmtree(offload_dir, ignore_errors=True)
        print("✅ Done")


if __name__ == "__main__":
    transcribe_videos_chunked()
