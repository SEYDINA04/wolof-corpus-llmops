"""
hf_dataset_upload.py
--------------------
Pushes a multispeaker TTS dataset to the Hugging Face Hub.

Each example on the Hub will contain:
    audio         – Audio feature (decoded waveform + sampling rate + path)
    speaker       – speaker name
    gender        – speaker gender  (ClassLabel: female / male)
    transcription – utterance text

Requirements
------------
pip install -r requirements.txt

Usage
-----
python hf_dataset_upload.py --repo your-hf-username/your-dataset-name --root /path/to/dataset --train-size 100 --private

Authentication
--------------
    Log in once with:   huggingface-cli login
    Or set the env var: HUGGINGFACE_TOKEN=hf_...
"""

import argparse
import os
import sys
import csv
from pathlib import Path

GENDER_LABELS = ["female", "male"]

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push a multispeaker TTS dataset to the Hugging Face Hub."
    )
    parser.add_argument(
        "--repo", required=True,
        help="Hub repository ID, e.g. 'username/my-tts-dataset'",
    )
    parser.add_argument(
        "--root", default=".",
        help="Path to the dataset root directory (default: current directory)",
    )
    parser.add_argument(
        "--token", default=None,
        help="Hugging Face token (falls back to HUGGINGFACE_TOKEN env var, then cached login)",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create the repository as private",
    )
    parser.add_argument(
        "--train-size", type=int, default=100,
        help="Number of samples to use as the train split (default: 100); remainder goes to test",
    )
    parser.add_argument(
        "--metadata", default="metadata.csv",
        help="Name of the pipe-separated metadata file (default: metadata.csv)",
    )
    parser.add_argument(
        "--num-shards", type=int, default=None,
        help="Number of Parquet shards to upload (useful for large datasets)",
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_imports() -> None:
    missing = []
    for pkg in ("datasets", "huggingface_hub", "soundfile"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        sys.exit(
            f"[ERROR] Missing packages: {', '.join(missing)}\n"
            f"Install with: pip install {' '.join(missing)}"
        )


def resolve_token(args: argparse.Namespace) -> str | None:
    return args.token or os.environ.get("HUGGINGFACE_TOKEN")


def load_metadata(root: Path, metadata_filename: str) -> list[dict]:
    """Load the pipe-separated metadata.csv and return a list of row dicts."""
    meta_path = root / metadata_filename
    if not meta_path.exists():
        sys.exit(
            f"[ERROR] Metadata file not found: {meta_path}\n"
            f"Run create_metadata.py first to generate it."
        )

    rows = []
    with meta_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            rows.append(row)

    if not rows:
        sys.exit("[ERROR] Metadata file is empty.")

    print(f"[INFO] Loaded {len(rows)} rows from {meta_path}")
    return rows


def build_hf_dataset(rows: list[dict], root: Path):
    """Convert metadata rows to a Hugging Face Dataset with an Audio column."""
    from datasets import Dataset, Audio, ClassLabel, Value, Features, Sequence

    # Resolve absolute audio paths (needed by the Audio feature).
    # filename now holds just the bare name (e.g. "afy_001.wav"),
    # so we reconstruct the full path from wavs/<speaker>/<filename>.
    abs_audio_paths = []
    missing = []
    for row in rows:
        wav_path = root / "wavs" / row["filename"]
        if not wav_path.exists():
            missing.append(str(wav_path))
        abs_audio_paths.append(str(wav_path))

    if missing:
        print(f"[WARNING] {len(missing)} audio file(s) not found on disk:")
        for p in missing[:10]:
            print(f"  {p}")
        if len(missing) > 10:
            print(f"  … and {len(missing) - 10} more")

    # Build column lists
    data = {
        "audio":         abs_audio_paths,
        "speaker":       [row["speaker"]       for row in rows],
        "gender":        [row["gender"]        for row in rows],
        "transcription": [row["transcription"] for row in rows],
    }

    # Define features explicitly so gender is stored as a ClassLabel
    features = Features({
        "audio":         Audio(sampling_rate=None),   # keeps native sample rate
        "speaker":       Value("string"),
        "gender":        ClassLabel(names=GENDER_LABELS),
        "transcription": Value("string"),
    })

    dataset = Dataset.from_dict(data, features=features)
    return dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    check_imports()

    root = Path(args.root).resolve()
    if not root.is_dir():
        sys.exit(f"[ERROR] Dataset root not found: {root}")

    token = resolve_token(args)

    # ── Load metadata ──────────────────────────────────────────────────────
    rows = load_metadata(root, args.metadata)

    # ── Build dataset ──────────────────────────────────────────────────────
    print("[INFO] Building Hugging Face Dataset …")
    dataset = build_hf_dataset(rows, root)
    print(f"[INFO] Dataset: {dataset}")

    # ── Train/test split ───────────────────────────────────────────────────
    from datasets import DatasetDict

    total = len(dataset)
    train_size = args.train_size

    if train_size >= total:
        sys.exit(
            f"[ERROR] --train-size ({train_size}) must be less than "
            f"the total number of samples ({total})."
        )

    # Shuffle with a fixed seed for reproducibility, then slice
    dataset = dataset.shuffle(seed=42)
    splits = DatasetDict({
        "train": dataset.select(range(train_size)),
        "test":  dataset.select(range(train_size, total)),
    })
    print(f"[INFO] Split → train: {len(splits['train'])}, test: {len(splits['test'])}")

    # ── Push to Hub ────────────────────────────────────────────────────────
    print(f"[INFO] Pushing to Hub: {args.repo}  (private={args.private}) …")

    push_kwargs = dict(
        repo_id=args.repo,
        private=args.private,
        token=token,
    )
    if args.num_shards is not None:
        # Apply the same shard count to every split
        push_kwargs["num_shards"] = {split: args.num_shards for split in splits}

    splits.push_to_hub(**push_kwargs)

    print(f"[INFO] Done! Dataset available at: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()