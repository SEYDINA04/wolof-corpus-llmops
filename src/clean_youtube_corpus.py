# clean_youtube_corpus.py

"""
Nettoyer les transcriptions YouTube (Whisper/Whosper) pour le corpus wolof.

Étapes:
  1. Charger les transcriptions JSON (video_id, text, ...)
  2. Effondrer les répétitions (boucles d'hallucination Whisper)
  3. Découper en phrases courtes (ponctuation si présente, sinon par paquets de mots)
  4. Supprimer les doublons (exacts, globaux)
  5. Sauvegarder au format HuggingFace: {"text": ..., "sources": [...]}

Usage:
    python clean_youtube_corpus.py
"""

import json
import re
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------
INPUT_DIR = Path("data/youtube_transcriptions_wolof")
OUTPUT_FILE = Path("data/youtube_transcriptions_wolof/merged_transcriptions.jsonl")

MIN_WORDS = 3          # phrase minimale (mots)
MAX_WORDS = 18         # phrase maximale -> on coupe au-delà
MAX_REPEAT_PHRASE = 6  # taille max d'un motif répété à effondrer

SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


def collapse_repetitions(words: List[str], max_phrase: int = MAX_REPEAT_PHRASE) -> List[str]:
    """Effondrer les motifs répétés consécutivement.

    Ex: ["a","b","a","b","a","b"] -> ["a","b"]
    Gère les boucles d'hallucination Whisper.
    """
    result: List[str] = []
    i = 0
    n = len(words)
    while i < n:
        collapsed = False
        # Essayer les motifs du plus long au plus court
        max_len = min(max_phrase, (n - i) // 2)
        for plen in range(max_len, 0, -1):
            phrase = words[i:i + plen]
            reps = 1
            j = i + plen
            while j + plen <= n and words[j:j + plen] == phrase:
                reps += 1
                j += plen
            if reps >= 2:
                result.extend(phrase)  # garder une seule copie
                i = j
                collapsed = True
                break
        if not collapsed:
            result.append(words[i])
            i += 1
    return result


def clean_text(text: str) -> str:
    """Normaliser espaces + effondrer les répétitions de mots."""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = collapse_repetitions(words)
    return " ".join(words)


def is_valid_sentence(text: str) -> bool:
    """Garder seulement les phrases avec assez de mots 'réels' (>= 2 lettres)."""
    real_words = [w for w in text.split() if len(re.sub(r"[^\w]", "", w)) >= 2]
    return len(real_words) >= MIN_WORDS


def chunk_by_words(words: List[str], max_words: int = MAX_WORDS) -> List[str]:
    """Découper une liste de mots en paquets de max_words.

    Un dernier paquet trop court est fusionné avec le précédent.
    """
    chunks = []
    for k in range(0, len(words), max_words):
        chunk = words[k:k + max_words]
        if len(chunk) < MIN_WORDS and chunks:
            chunks[-1] += " " + " ".join(chunk)  # fusionner le reliquat
        else:
            chunks.append(" ".join(chunk))
    return chunks


def split_to_sentences(text: str) -> List[str]:
    """Découper un texte nettoyé en phrases courtes."""
    sentences: List[str] = []
    # 1. Sur la ponctuation si présente
    parts = SENTENCE_PATTERN.split(text)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        words = part.split()
        # 2. Si trop long (pas de ponctuation), couper par paquets de mots
        if len(words) > MAX_WORDS:
            sentences.extend(chunk_by_words(words))
        elif len(words) >= MIN_WORDS:
            sentences.append(part)
    # 3. Filtrer les fragments parasites (déchets à 1 lettre, etc.)
    return [s for s in sentences if is_valid_sentence(s)]


def main():
    print("=" * 60)
    print("NETTOYAGE CORPUS YOUTUBE WOLOF")
    print("=" * 60 + "\n")

    json_files = sorted(
        f for f in INPUT_DIR.glob("*.json")
        if f.name != "transcriptions_summary.json"
    )
    print(f"Fichiers trouvés: {len(json_files)}\n")

    all_items = []
    seen = set()          # dédup global (texte normalisé)
    raw_count = 0

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        video_id = data.get("video_id", jf.stem)
        text = data.get("text", "")
        source = f"https://www.youtube.com/watch?v={video_id}"

        cleaned = clean_text(text)
        sentences = split_to_sentences(cleaned)

        for sent in sentences:
            raw_count += 1
            key = sent.lower().strip()
            if key in seen:
                continue          # doublon global
            seen.add(key)
            all_items.append({"text": sent, "sources": [source]})

        print(f"  {video_id}: {len(sentences)} phrases extraites")

    # Sauvegarde
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Stats
    total_words = sum(len(it["text"].split()) for it in all_items)
    print("\n" + "=" * 60)
    print("STATISTIQUES")
    print("=" * 60)
    print(f"  - Phrases avant dédup : {raw_count}")
    print(f"  - Phrases finales     : {len(all_items)}")
    print(f"  - Doublons supprimés  : {raw_count - len(all_items)}")
    print(f"  - Total mots          : {total_words}")
    if all_items:
        lengths = [len(it["text"].split()) for it in all_items]
        print(f"  - Longueur min/max    : {min(lengths)}/{max(lengths)} mots")

    print(f"\nSauvegardé: {OUTPUT_FILE}")
    print("\nÉCHANTILLON:")
    for i, it in enumerate(all_items[:5], 1):
        print(f"  {i}. {it['text']}")


if __name__ == "__main__":
    main()
