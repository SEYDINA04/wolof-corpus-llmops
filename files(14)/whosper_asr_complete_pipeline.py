"""
PIPELINE ASR COMPLET: Whosper-large-v2 avec Confidence, Post-processing, WER
Babacar Ndao | GalsenAI Corpus Wolof
Prêt pour Google Colab GPU T4
"""

import json
import numpy as np
import librosa
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from transformers import AutoModelForSpeech2Seq2Seq, AutoProcessor
from peft import PeftModel

# ==================== ÉTAPE 1: CHUNK & STRIDE ====================

class AudioChunker:
    """Découpe l'audio avec overlap configurable."""

    def __init__(self, chunk_duration: int = 20, stride: int = 18):
        """
        Args:
            chunk_duration: durée du chunk en secondes (défaut 20s, était 12s)
            stride: distance entre starts de chunks (défaut 18s → 2s overlap)
        """
        self.chunk_duration = chunk_duration
        self.stride = stride
        self.overlap = chunk_duration - stride

    def chunk_audio(self, audio: np.ndarray, sr: int = 16000) -> List[np.ndarray]:
        """Découpe l'audio en chunks avec overlap."""
        chunk_samples = self.chunk_duration * sr
        stride_samples = self.stride * sr

        chunks = []
        for start in range(0, len(audio), stride_samples):
            end = start + chunk_samples
            if end > len(audio):
                # Padding du dernier chunk si nécessaire
                chunk = np.zeros(chunk_samples)
                chunk[:len(audio)-start] = audio[start:len(audio)]
            else:
                chunk = audio[start:end]
            chunks.append(chunk)

        return chunks

    def get_chunk_metadata(self) -> Dict:
        """Retourne config pour logs."""
        return {
            "chunk_duration_s": self.chunk_duration,
            "stride_s": self.stride,
            "overlap_s": self.overlap,
            "overlap_percentage": (self.overlap / self.chunk_duration) * 100
        }


# ==================== ÉTAPE 2: CONFIDENCE EXTRACTION ====================

class WhosperTranscriber:
    """Whosper-large-v2 avec extraction confidence scores."""

    def __init__(self, model_id: str = "CAYTU/whosper-large-v2"):
        """Charge le modèle Whosper avec PEFT adapter."""
        print(f"📥 Chargement {model_id}...")

        # Modèle base
        base_model_id = "openai/whisper-large-v2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForSpeech2Seq2Seq.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(device)

        # Charger PEFT adapter
        self.model = PeftModel.from_pretrained(self.model, model_id)
        self.model = self.model.to(device)

        # Processor
        self.processor = AutoProcessor.from_pretrained(base_model_id)
        self.device = device
        self.dtype = dtype

        print(f"✅ Modèle chargé sur {device}")

    def transcribe_with_confidence(self, audio: np.ndarray, sr: int = 16000) -> Dict:
        """
        Transcrit et extrait confidence score.
        Retourne: {text, confidence, log_probs}
        """
        # Prétraiter audio
        inputs = self.processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            return_attention_mask=True
        ).to(self.device)

        input_features = inputs.input_features.to(self.dtype)
        attention_mask = inputs.get("attention_mask")

        # Générer avec output_scores pour confidence
        with torch.no_grad():
            outputs = self.model.generate(
                inputs=input_features,
                attention_mask=attention_mask,
                max_new_tokens=300,
                task="transcribe",
                return_dict_in_generate=True,
                output_scores=True
            )

        # Décoder texte
        text = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]

        # Calculer confidence = moyenne des log-probs
        if outputs.scores:
            # Log-probs par token
            log_probs = []
            for i, scores in enumerate(outputs.scores):
                token_logprobs = torch.log_softmax(scores, dim=-1)
                # Probabilité du token généré
                top_logprob = torch.max(token_logprobs, dim=-1).values[0]
                log_probs.append(top_logprob.item())

            confidence = np.exp(np.mean(log_probs))  # Moyenne exponentiée
            confidence = float(np.clip(confidence, 0, 1))  # Clamp 0-1
        else:
            confidence = 0.5  # Fallback

        return {
            "text": text.strip(),
            "confidence": confidence,
            "log_probs": log_probs if outputs.scores else []
        }


# ==================== ÉTAPE 3: POST-PROCESSING ====================

class PostProcessor:
    """Nettoyage et fusion des transcriptions."""

    @staticmethod
    def deduplicate_repeats(text: str, min_repeat_count: int = 3) -> str:
        """Supprime les répétitions de mots/phrases."""
        words = text.split()
        cleaned = []
        i = 0

        while i < len(words):
            word = words[i]
            # Compter répétitions consécutives
            repeat_count = 1
            while i + repeat_count < len(words) and words[i + repeat_count] == word:
                repeat_count += 1

            # Garder si <3 répétitions, sinon garder 1×
            if repeat_count < min_repeat_count:
                cleaned.extend([word] * repeat_count)
            else:
                cleaned.append(word)

            i += repeat_count

        return " ".join(cleaned)

    @staticmethod
    def filter_by_confidence(transcriptions: List[Dict], threshold: float = 0.5) -> List[Dict]:
        """Filtre les chunks avec confidence < threshold."""
        filtered = []
        for trans in transcriptions:
            if trans["confidence"] >= threshold:
                filtered.append(trans)
            else:
                # Log low-confidence pour inspection
                print(f"⚠️  Confidence basse ({trans['confidence']:.2f}): {trans['text'][:50]}...")

        return filtered

    @staticmethod
    def merge_overlapping_chunks(
        chunks: List[Dict],
        overlap_duration: int = 2
    ) -> str:
        """
        Fusionne chunks chevauchants en détectant et supprimant duplication.
        Heuristique: si N dernier mots du chunk N matchent N premiers du chunk N+1,
        c'est une zone overlap → garder 1×.
        """
        if not chunks:
            return ""

        merged_text = chunks[0]["text"]

        for i in range(1, len(chunks)):
            current_text = chunks[i]["text"]

            # Essayer de détecter overlap par matching de mots
            prev_words = merged_text.split()
            curr_words = current_text.split()

            # Chercher si N derniers mots de prev matchent N premiers de curr
            overlap_found = False
            for overlap_len in range(min(5, len(prev_words), len(curr_words)), 0, -1):
                if prev_words[-overlap_len:] == curr_words[:overlap_len]:
                    # Overlap détecté: ajouter seulement le reste
                    merged_text += " " + " ".join(curr_words[overlap_len:])
                    overlap_found = True
                    break

            if not overlap_found:
                # Pas d'overlap détecté: ajouter normalement
                merged_text += " " + current_text

        return merged_text.strip()

    @staticmethod
    def segment_sentences(text: str) -> List[str]:
        """Segmente le texte en phrases (simple heuristique pour Wolof)."""
        # Marqueurs Wolof pour fin de phrase (simplifié)
        delimiters = ['.', '!', '?', '\\n']

        sentences = [text]
        for delim in delimiters:
            new_sentences = []
            for sent in sentences:
                new_sentences.extend(sent.split(delim))
            sentences = new_sentences

        return [s.strip() for s in sentences if s.strip()]


# ==================== ÉTAPE 4: WER MEASUREMENT ====================

class WERCalculator:
    """Calcule Word Error Rate (WER) entre référence et hypothèse."""

    @staticmethod
    def compute_wer(reference: str, hypothesis: str) -> float:
        """
        Calcule WER = (S + D + I) / N
        S = substitutions, D = deletions, I = insertions, N = mots référence
        """
        ref_words = reference.split()
        hyp_words = hypothesis.split()

        # Dynamic programming pour edit distance
        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))

        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = 1 + min(
                        d[i-1][j],      # deletion
                        d[i][j-1],      # insertion
                        d[i-1][j-1]     # substitution
                    )

        wer = d[len(ref_words)][len(hyp_words)] / len(ref_words) if len(ref_words) > 0 else 0
        return float(wer)

    @staticmethod
    def compute_cer(reference: str, hypothesis: str) -> float:
        """Character Error Rate (CER) — même logique mais au niveau caractère."""
        ref_chars = list(reference.replace(" ", ""))
        hyp_chars = list(hypothesis.replace(" ", ""))

        d = np.zeros((len(ref_chars) + 1, len(hyp_chars) + 1))
        for i in range(len(ref_chars) + 1):
            d[i][0] = i
        for j in range(len(hyp_chars) + 1):
            d[0][j] = j

        for i in range(1, len(ref_chars) + 1):
            for j in range(1, len(hyp_chars) + 1):
                if ref_chars[i-1] == hyp_chars[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])

        cer = d[len(ref_chars)][len(hyp_chars)] / len(ref_chars) if len(ref_chars) > 0 else 0
        return float(cer)


# ==================== PIPELINE COMPLET ====================

def run_asr_pipeline(
    audio_path: str,
    output_dir: str = "/tmp/wolof_transcriptions",
    chunk_duration: int = 20,
    stride: int = 18,
    confidence_threshold: float = 0.5
) -> Dict:
    """
    Pipeline complet: charge audio → chunks → transcribe + confidence →
    post-process → export JSON.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"🎤 PIPELINE ASR: {Path(audio_path).name}")
    print(f"{'='*70}")

    # Charger audio
    print(f"📊 Chargement audio...")
    audio, sr = librosa.load(audio_path, sr=16000)
    duration_s = len(audio) / sr
    print(f"   ✅ {duration_s:.1f}s @ {sr} Hz")

    # Étape 1: Chunking
    print(f"\n🔹 ÉTAPE 1: CHUNKING")
    chunker = AudioChunker(chunk_duration=chunk_duration, stride=stride)
    chunks = chunker.chunk_audio(audio, sr=sr)
    metadata = chunker.get_chunk_metadata()
    print(f"   Chunks: {len(chunks)}")
    print(f"   Config: {metadata}")

    # Étape 2: Transcription + Confidence
    print(f"\n🔹 ÉTAPE 2: TRANSCRIPTION + CONFIDENCE")
    transcriber = WhosperTranscriber()
    all_transcriptions = []

    for i, chunk in enumerate(chunks):
        result = transcriber.transcribe_with_confidence(chunk, sr=sr)
        all_transcriptions.append({
            "chunk_id": i,
            **result
        })

        if (i + 1) % 50 == 0:
            avg_conf = np.mean([t["confidence"] for t in all_transcriptions])
            print(f"   [{i+1}/{len(chunks)}] Confidence moyenne: {avg_conf:.3f}")

    # Étape 3: Post-processing
    print(f"\n🔹 ÉTAPE 3: POST-PROCESSING")

    # 3a. Filtrer par confidence
    print(f"   Avant filtre: {len(all_transcriptions)} chunks")
    filtered = PostProcessor.filter_by_confidence(all_transcriptions, threshold=confidence_threshold)
    print(f"   Après filtre (conf>{confidence_threshold}): {len(filtered)} chunks")

    # 3b. Merger chunks
    merged_text = PostProcessor.merge_overlapping_chunks(filtered)
    print(f"   Texte fusionné: {len(merged_text)} caractères")

    # 3c. Dédupliquer
    deduplicated = PostProcessor.deduplicate_repeats(merged_text)
    print(f"   Après dédup: {len(deduplicated)} caractères")

    # 3d. Segmenter (optionnel)
    sentences = PostProcessor.segment_sentences(deduplicated)
    print(f"   Phrases détectées: {len(sentences)}")

    # Étape 4: Export + WER (si référence disponible)
    print(f"\n🔹 ÉTAPE 4: EXPORT + WER")

    output_data = {
        "video_id": Path(audio_path).stem,
        "duration_s": duration_s,
        "num_chunks": len(chunks),
        "chunk_config": metadata,
        "transcription_raw": merged_text,
        "transcription_cleaned": deduplicated,
        "num_sentences": len(sentences),
        "sentences": sentences,
        "chunk_details": [
            {
                "chunk_id": t["chunk_id"],
                "text": t["text"][:100],  # First 100 chars
                "confidence": round(t["confidence"], 3)
            }
            for t in filtered
        ],
        "quality_metrics": {
            "avg_confidence": round(np.mean([t["confidence"] for t in filtered]), 3),
            "chunks_low_confidence": len(all_transcriptions) - len(filtered),
            "total_characters": len(deduplicated)
        }
    }

    # Sauvegarder JSON
    output_file = output_dir / f"{Path(audio_path).stem}_asr_output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"   💾 Sauvegardé: {output_file}")
    print(f"   📊 Caractères finaux: {len(deduplicated)}")
    print(f"   📈 Confidence moyenne: {output_data['quality_metrics']['avg_confidence']}")

    return output_data


def compare_with_reference(hypothesis: str, reference: str = None) -> Dict:
    """
    Calcule WER/CER si référence disponible.
    USAGE: reference = "ndeysaan coyy ati na Cëy àddina jamono day"
    """
    if reference is None:
        print("⚠️  Pas de référence fournie → WER non calculable")
        return {}

    print(f"\n🔹 COMPARAISON AVEC RÉFÉRENCE")
    print(f"   Référence: {reference[:80]}...")
    print(f"   Hypothèse: {hypothesis[:80]}...")

    wer = WERCalculator.compute_wer(reference, hypothesis)
    cer = WERCalculator.compute_cer(reference, hypothesis)

    print(f"\n   📊 WER (Word Error Rate): {wer:.1%}")
    print(f"   📊 CER (Char Error Rate): {cer:.1%}")

    return {"wer": wer, "cer": cer}


# ==================== EXEMPLE USAGE ====================

if __name__ == "__main__":
    """
    # Pour Google Colab:

    # 1. Importer et exécuter ce script
    # 2. Vérifier que Whosper est installé:
    #    !pip install -q git+https://github.com/sudoping01/whosper.git

    # 3. Exécuter pipeline sur une vidéo

    result = run_asr_pipeline(
        audio_path="/tmp/wolof_videos/_eqpwNEXEzs.mp4",
        chunk_duration=20,  # 20s chunks (was 12s)
        stride=18,          # 18s stride → 2s overlap (was 11s → 1s)
        confidence_threshold=0.5
    )

    # 4. Optionnel: comparer avec référence Wolof
    reference = "ndeysaan coyy ati na Cëy àddina jamono day dara"
    compare_with_reference(result["transcription_cleaned"], reference)
    """

    print("✅ Pipeline ASR importé. Utilise run_asr_pipeline() dans Colab.")
