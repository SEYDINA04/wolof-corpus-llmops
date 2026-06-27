# push_wolof_to_hf.py

"""
Push le corpus wolof-online merged vers HuggingFace Hub.
Met à jour le fichier Parquet d'un repo EXISTANT.

Usage:
    python push_wolof_to_hf.py --parquet wolof_centalized_corpus/data/train-00000-of-00001.parquet
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Push corpus wolof vers Hugging Face Hub"
    )
    parser.add_argument(
        "--repo",
        default="galsenai/wolof_centalized_corpus",
        help="Hub repository ID"
    )
    parser.add_argument(
        "--parquet",
        required=True,
        help="Path to the parquet file"
    )
    parser.add_argument(
        "--commit-message",
        default="Add wolof-online.com corpus (1,182 phrases, 35k words)",
        help="Commit message"
    )
    parser.add_argument(
        "--token-file",
        default=None,
        help="Path to file containing HF token"
    )
    return parser.parse_args()


def load_token(token_file=None):
    """Load token from file or prompt user"""
    from huggingface_hub import login

    if token_file:
        token_path = Path(token_file)
        if token_path.exists():
            print(f" Loading token from: {token_file}")
            token = token_path.read_text().strip()
            login(token=token)
            return token

    # Sinon, prompt interactif
    print(" Please enter your HuggingFace token:")
    login()
    return None


def main():
    print("="*60)
    print("PUSH CORPUS WOLOF VERS HUGGINGFACE")
    print("="*60 + "\n")

    args = parse_args()

    # Check dependencies
    try:
        from huggingface_hub import HfApi
        import pandas as pd
    except ImportError as e:
        sys.exit(f"[ERROR] Missing package: {e.name}\nRun: pip install huggingface_hub pandas pyarrow")

    # Authenticate
    print("Authentification HuggingFace...")
    load_token(args.token_file)
    print("Authentifié\n")

    # Load parquet
    parquet_path = Path(args.parquet).resolve()
    if not parquet_path.exists():
        sys.exit(f"[ERROR] File not found: {parquet_path}")

    print(f"Loading: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f" {len(df):,} items loaded\n")

    # Display sample
    print("Sample:")
    print(f"  text: {df['text'].iloc[0][:80]}...")
    print(f"  sources: {df['sources'].iloc[0]}\n")

    # Upload file using HfApi (pas push_to_hub qui essaie de créer le repo)
    print(f"Uploading to {args.repo}...")

    try:
        api = HfApi()

        # Upload le fichier directement (ne crée PAS le repo)
        api.upload_file(
            path_or_fileobj=str(parquet_path),
            path_in_repo="data/train-00000-of-00001.parquet",
            repo_id=args.repo,
            repo_type="dataset",
            commit_message=args.commit_message
        )

        print(f"\nSUCCESS!")
        print(f" https://huggingface.co/datasets/{args.repo}")

    except Exception as e:
        print(f"\n ERROR: {e}")
        print("\n Verify:")
        print("  - Repo exists: galsenai/wolof_centalized_corpus")
        sys.exit(1)


if __name__ == "__main__":
    main()
