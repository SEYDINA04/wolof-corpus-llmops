# src/push_unified_to_hf.py
"""
Push du corpus unifié (606 456 exemples) vers le dataset HF existant
galsenai/wolof_centalized_corpus, SANS écraser les données existantes :
le corpus unifié = corpus HF actuel (304 762) + corpus ingéré (424 119),
déduplication exacte appliquée (fusion des listes `sources`).

Met à jour 2 fichiers du repo :
  - data/train-00000-of-00001.parquet
  - README.md (métadonnées dataset_info)

Sécurité :
  - utilise HfApi.upload_file (pas de git) -> n'ajoute jamais les fichiers token
  - --dry-run pour tout valider sans rien envoyer

Usage :
  .venv/bin/python push_unified_to_hf.py --dry-run
  .venv/bin/python push_unified_to_hf.py            # push réel
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = "galsenai/wolof_centalized_corpus"
HERE = Path(__file__).parent
REPO_DIR = HERE / "wolof_centalized_corpus"
PARQUET = REPO_DIR / "data" / "train-00000-of-00001.parquet"
README = REPO_DIR / "README.md"
TOKEN_FILE = REPO_DIR / "galsenai_lab_access_token.txt"
COMMIT_MSG = (
    "Add ingested HF datasets (17 sources) — corpus 304,762 -> 606,456 "
    "examples (deduplicated, sources merged)"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--token-file", default=str(TOKEN_FILE))
    args = ap.parse_args()

    from huggingface_hub import HfApi

    # --- garde-fous ---
    if not PARQUET.exists():
        sys.exit(f"[ERROR] parquet introuvable: {PARQUET}")
    df = pd.read_parquet(PARQUET)
    n = len(df)
    print(f"Parquet à pousser : {PARQUET}")
    print(f"  -> {n:,} exemples, colonnes={list(df.columns)}")
    if n < 606000:
        sys.exit(f"[ERROR] sécurité: {n:,} < 606 000 attendus, abort.")

    token = Path(args.token_file).read_text().strip()
    api = HfApi(token=token)
    who = api.whoami()
    print(f"Authentifié : {who.get('name')}  orgs={[o['name'] for o in who.get('orgs', [])]}")

    # vérif anti-perte : les textes HF actuels sont-ils tous présents ?
    from huggingface_hub import hf_hub_download
    cur = pd.read_parquet(hf_hub_download(REPO, "data/train-00000-of-00001.parquet", repo_type="dataset", token=token))
    kcur = set(cur["text"].fillna("").str.lower().str.strip()); kcur.discard("")
    knew = set(df["text"].fillna("").str.lower().str.strip()); knew.discard("")
    missing = kcur - knew
    print(f"Contrôle anti-perte : {len(kcur):,} textes HF actuels, manquants dans le nouveau = {len(missing):,}")
    if missing:
        sys.exit(f"[ERROR] {len(missing):,} textes HF seraient perdus, abort.")
    print(f"  OK — +{len(knew - kcur):,} nouveaux textes ajoutés")

    if args.dry_run:
        print("\n[DRY-RUN] Aucune donnée envoyée. Fichiers qui seraient poussés :")
        print(f"  - data/train-00000-of-00001.parquet ({PARQUET.stat().st_size:,} o)")
        print(f"  - README.md")
        print(f"  commit: {COMMIT_MSG}")
        return

    print("\nUpload README.md...")
    api.upload_file(path_or_fileobj=str(README), path_in_repo="README.md",
                    repo_id=REPO, repo_type="dataset", commit_message=COMMIT_MSG)
    print("Upload parquet...")
    api.upload_file(path_or_fileobj=str(PARQUET), path_in_repo="data/train-00000-of-00001.parquet",
                    repo_id=REPO, repo_type="dataset", commit_message=COMMIT_MSG)
    print(f"\nSUCCESS  -> https://huggingface.co/datasets/{REPO}")


if __name__ == "__main__":
    main()
