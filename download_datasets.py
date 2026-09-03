#!/usr/bin/env python3
"""
Dataset Downloader for SWE Agent Pretraining & SFT Matrix.
Downloads full raw datasets directly into target directory paths (bypassing HF cache symlinks).

Usage:
    # List available dataset keys:
    python download_datasets.py --list

    # Download a specific dataset with custom workers:
    python download_datasets.py --dataset dolma3.5-pool --workers 24
    python download_datasets.py --dataset starcoderdata --token <YOUR_TOKEN>

    # Download all datasets sequentially (from smallest to largest):
    python download_datasets.py --workers 16
"""

import argparse
import os
import sys
import subprocess
from huggingface_hub import snapshot_download

DEFAULT_WORKERS = 8

# Ordered from smallest to largest volume
DATASET_SPECS = [
    # Git Repositories
    {
        "key": "swe-atlas",
        "name": "SWE-Atlas",
        "type": "git",
        "url": "https://github.com/scaleapi/SWE-Atlas.git",
        "dir": "swe-atlas",
        "size_est": "0.7B tokens / ~50 MB repo",
        "gated": False,
        "allow_patterns": None,
    },
    {
        "key": "fvapps",
        "name": "Lean4 Verification Library (fvapps)",
        "type": "git",
        "url": "https://github.com/quinn-dougherty/fvapps.git",
        "dir": "fvapps",
        "size_est": "1.0B tokens / ~100 MB repo",
        "gated": False,
        "allow_patterns": None,
    },
    # Hugging Face Datasets
    {
        "key": "swe-bench-multimodal",
        "name": "SWE-bench Multimodal Logic",
        "type": "hf",
        "repo_id": "princeton-nlp/SWE-bench_Multimodal",
        "dir": "swe-bench-multimodal",
        "size_est": "0.4B tokens / ~500 MB",
        "gated": False,
        "allow_patterns": None,
    },
    {
        "key": "swe-bench",
        "name": "DeepSWE / Canonical SWE-bench",
        "type": "hf",
        "repo_id": "SWE-bench/SWE-bench",
        "dir": "swe-bench",
        "size_est": "0.5B tokens / ~1.2 GB",
        "gated": False,
        "allow_patterns": None,
    },
    {
        "key": "minerva-math",
        "name": "Minerva Math (Symbolic / Reasoning)",
        "type": "hf",
        "repo_id": "math-ai/AutoMathText",
        "dir": "minerva_math",
        "size_est": "149 GB / Mathematical proofs & reasoning traces",
        "gated": False,
        "allow_patterns": None,
    },
    {
        "key": "tulu-3-sft-mixture",
        "name": "AI2 Tülu 3 Instruction Mix",
        "type": "hf",
        "repo_id": "allenai/tulu-3-sft-mixture",
        "dir": "tulu-3-sft-mixture",
        "size_est": "0.5B tokens / ~2.5 GB",
        "gated": False,
        "allow_patterns": None,
    },
    {
        "key": "lean4-mathlib",
        "name": "Lean4 Mathlib Symbolic Proofs",
        "type": "hf",
        "repo_id": "phanerozoic/Lean4-Mathlib",
        "dir": "lean4-mathlib",
        "size_est": "1.0B tokens / ~4.0 GB",
        "gated": False,
        "allow_patterns": None,
    },
    {
        "key": "starcoderdata",
        "name": "Function-Aware FIM Subsplit (StarCoder)",
        "type": "hf",
        "repo_id": "bigcode/starcoderdata",
        "dir": "starcoderdata",
        "size_est": "3.0B tokens / ~289 GB (full pool)",
        "gated": True,
        "gated_url": "https://huggingface.co/datasets/bigcode/starcoderdata",
        "allow_patterns": None,
    },
    {
        "key": "nemotron-pretraining-code-v3",
        "name": "Nemotron-Pretraining-Code-v3",
        "type": "hf",
        "repo_id": "nvidia/Nemotron-Pretraining-Code-v3",
        "dir": "nemotron-pretraining-code-v3",
        "size_est": "20.0B tokens / ~7.66 GB",
        "gated": False,
        "allow_patterns": None,
    },
    {
        "key": "essential-web",
        "name": "Essential-Web-v1.0 (API Docs & Architecture)",
        "type": "hf",
        "repo_id": "EssentialAI/essential-web-v1.0",
        "dir": "essential-web",
        "size_est": "40.0B tokens / Pre-cached",
        "gated": False,
        "allow_patterns": None,
    },
    {
        "key": "dolma3.5-pool",
        "name": "Dolma 3.5 Code Split",
        "type": "hf",
        "repo_id": "allenai/dolma3.5_pool",
        "dir": "dolma3.5-pool",
        "size_est": "75.0B tokens / Code sub-pool",
        "gated": False,
        # Fast filter for code partitions in dolma3.5_pool
        "allow_patterns": ["dolma_code/*", "common_pile_code/*", "swallow-code/*", "the-stack-v2/*", "*.md", "*.gitattributes"],
    },
]


def resolve_hf_token(cli_token: str | None = None) -> str | None:
    """Resolves HF token from CLI argument, environment variables, or huggingface-cli cache."""
    if cli_token:
        return cli_token
    
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token

    # Check local huggingface-cli authentication cache (~/.cache/huggingface/token)
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            return token
    except Exception:
        pass

    return None


def resolve_base_dir(cli_dir: str | None = None) -> str:
    """Resolves target base directory, raising an explicit error if not set."""
    base_dir = cli_dir or os.environ.get("DATASETS_DIR")
    if not base_dir:
        print("[ERROR] No target datasets storage directory specified!", file=sys.stderr)
        print("Please provide a destination directory via either:", file=sys.stderr)
        print("  1. CLI argument:        --base-dir /path/to/datasets (or -b)", file=sys.stderr)
        print("  2. Environment variable: export DATASETS_DIR=/path/to/datasets", file=sys.stderr)
        sys.exit(1)

    return os.path.abspath(os.path.expanduser(base_dir))


def download_item(item, base_dir, token, max_workers):
    name = item["name"]
    target_dir = os.path.join(base_dir, item["dir"])

    print(f"\n[{name}] ({item['size_est']})")
    print(f"Destination: {target_dir}")
    print(f"Parallel Workers: {max_workers}")

    if item["type"] == "git":
        url = item["url"]
        if os.path.exists(target_dir):
            print("  -> Directory exists. Pulling latest updates...")
            subprocess.run(["git", "-C", target_dir, "pull"], check=False)
        else:
            print(f"  -> Cloning repository from {url}...")
            subprocess.run(["git", "clone", url, target_dir], check=True)
            print("  -> Clone complete.")

    elif item["type"] == "hf":
        repo_id = item["repo_id"]
        is_gated = item.get("gated", False)

        # Check token requirement for gated datasets
        if is_gated and not token:
            gated_url = item.get("gated_url", f"https://huggingface.co/datasets/{repo_id}")
            print(f"\n[ERROR] '{item['key']}' ({repo_id}) is a GATED dataset and requires authentication!", file=sys.stderr)
            print(f"To download this dataset:", file=sys.stderr)
            print(f"  1. Accept user license terms at: {gated_url}", file=sys.stderr)
            print(f"  2. Provide an authorized token via one of:", file=sys.stderr)
            print(f"     - CLI argument:        --token <YOUR_HF_TOKEN>", file=sys.stderr)
            print(f"     - Environment variable: export HF_TOKEN=\"<YOUR_HF_TOKEN>\"", file=sys.stderr)
            print(f"     - Login via CLI:        huggingface-cli login\n", file=sys.stderr)
            return False

        os.makedirs(target_dir, exist_ok=True)
        print(f"  -> Downloading raw files from Hugging Face: {repo_id} ...")
        try:
            kwargs = {
                "repo_id": repo_id,
                "repo_type": "dataset",
                "local_dir": target_dir,
                "token": token,
                "max_workers": max_workers,
                "resume_download": True,
            }
            if item.get("allow_patterns"):
                kwargs["allow_patterns"] = item["allow_patterns"]
                print(f"  -> Filtering partitions: {item['allow_patterns']}")

            snapshot_download(**kwargs)
            print(f"  -> Successfully downloaded raw files for {repo_id}")
            return True
        except Exception as e:
            print(f"  -> Error downloading {repo_id}: {e}", file=sys.stderr)
            return False


def main():
    parser = argparse.ArgumentParser(description="Download raw datasets for SWE Agent training.")
    parser.add_argument("--dataset", "-d", type=str, help="Download a specific dataset by key")
    parser.add_argument("--list", "-l", action="store_true", help="List all dataset keys and size estimates")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS, help="Number of parallel download workers (default: 8)")
    parser.add_argument("--base-dir", "-b", type=str, default=None, help="Target storage directory (or set DATASETS_DIR env var)")
    parser.add_argument("--token", "-t", type=str, default=None, help="Hugging Face API token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    if args.list:
        print("Available datasets (in download order from smallest to largest):")
        for item in DATASET_SPECS:
            gated_flag = " [GATED 🔒]" if item.get("gated") else ""
            print(f"  - {item['key']:<28} [{item['size_est']}] -> {item['name']}{gated_flag}")
        return

    base_dir = resolve_base_dir(args.base_dir)
    token = resolve_hf_token(args.token)

    os.makedirs(base_dir, exist_ok=True)

    if args.dataset:
        matched = [item for item in DATASET_SPECS if item["key"] == args.dataset]
        if not matched:
            print(f"Error: Dataset key '{args.dataset}' not found. Run with --list to see options.", file=sys.stderr)
            sys.exit(1)
        download_item(matched[0], base_dir, token, args.workers)
    else:
        print("============================================================")
        print("Downloading all datasets sequentially (smallest to largest)...")
        print(f"Target Directory: {base_dir}")
        print(f"Workers: {args.workers}")
        print("============================================================")
        for item in DATASET_SPECS:
            download_item(item, base_dir, token, args.workers)

    print("\nDone.")


if __name__ == "__main__":
    main()
