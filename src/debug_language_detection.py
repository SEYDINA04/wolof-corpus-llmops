# debug_language_detection.py
import json
import fasttext

model = fasttext.load_model("lid.176.bin")

with open("data/validated/n1_rejected_corpus.jsonl", 'r') as f:
    for i, line in enumerate(f):
        if i >= 3:  # Analyser seulement les 3 premiers
            break

        article = json.loads(line)
        text = article.get('full_content', '')[:500]  # Premiers 500 chars
        text_clean = text.replace('\n', ' ').strip()  # ← FIX: enlever les \n

        pred = model.predict(text_clean, k=3)  # Top 3 langues

        print(f"\n{'='*60}")
        print(f"Article {i+1}: {article.get('title', 'Sans titre')}")
        print(f"{'='*60}")
        print(f"Texte (extrait):\n{text}\n")
        print(f"Prédictions fastText:")
        for lang, prob in zip(pred[0], pred[1]):
            print(f"  - {lang.replace('__label__', '')}: {prob*100:.1f}%")
