#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download multilingual CLIP ViT-B-32 ONNX models and tokenizer files.

This uses the official sentence-transformers multilingual CLIP approach:
- Vision: Original clip-ViT-B-32 for encoding images
- Text: clip-ViT-B-32-multilingual-v1 aligned to vision model (supports 50+ languages)

The multilingual text model is trained to map text from multiple languages into
the same embedding space as the original CLIP vision encoder.
"""

import argparse
import logging
import os
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Model URLs - Multilingual CLIP using the official approach:
# - Vision: Original clip-ViT-B-32 for encoding images
# - Text: clip-ViT-B-32-multilingual-v1 aligned to the vision model (50+ languages)
# See: https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1
MODEL_URLS = {
    "vision": "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model.onnx",
    "text": "https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1/resolve/main/onnx/model.onnx",
    "tokenizer": "https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1/resolve/main/tokenizer.json",
    "vocab": "https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1/resolve/main/vocab.txt",
}

# Target filenames in models directory
TARGET_NAMES = {
    "vision": "clip-vit-b-32-multilingual-vision.onnx",
    "text": "clip-vit-b-32-multilingual-text.onnx",
    "tokenizer": "clip-tokenizer.json",
    "vocab": "clip-vocab.txt",
}


def download_file(url: str, dest_path: Path, desc: str = "Downloading"):
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with (
            open(dest_path, "wb") as f,
            tqdm(
                desc=desc,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        logging.info(
            f"✓ Downloaded: {dest_path.name} ({dest_path.stat().st_size / 1024 / 1024:.2f} MB)"
        )
        return True

    except requests.exceptions.RequestException as e:
        logging.error(f"✗ Failed to download {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def verify_file(path: Path, min_size_mb: float = 0.1):
    """Verify that a file exists and is not empty."""
    if not path.exists():
        logging.error(f"✗ File not found: {path}")
        return False

    size_mb = path.stat().st_size / 1024 / 1024
    if size_mb < min_size_mb:
        logging.error(f"✗ File too small ({size_mb:.2f} MB): {path}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download multilingual CLIP ViT-B-32 ONNX models (50+ languages)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing files, don't download",
    )
    args = parser.parse_args()

    # Determine models directory
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    logging.info(f"Models directory: {models_dir}")

    # Download or verify each file
    success = True
    for key, url in MODEL_URLS.items():
        target_path = models_dir / TARGET_NAMES[key]

        if args.verify_only:
            if verify_file(target_path):
                logging.info(f"✓ Verified: {target_path.name}")
            else:
                success = False
            continue

        # Check if file exists and skip if not forcing
        if target_path.exists() and not args.force:
            if verify_file(target_path):
                logging.info(
                    f"✓ Already exists: {target_path.name} (use --force to re-download)"
                )
                continue

        # Download the file
        logging.info(f"Downloading {key} model from {url}...")
        if not download_file(url, target_path, desc=f"Downloading {key}"):
            success = False
            continue

        # Verify download
        if not verify_file(target_path):
            success = False

    # Final verification
    if success:
        logging.info("\n✓ All CLIP models downloaded and verified successfully!")
        logging.info(
            "You can now select 'CLIP-ViT-B-32-multilingual' in PicFinder settings."
        )
        logging.info(
            "\nNote: For better multilingual support, install: pip install tokenizers"
        )
    else:
        logging.error("\n✗ Some downloads failed. Please check the errors above.")
        logging.error("Note: You may need to manually download the models from:")
        logging.error(
            "  Vision: https://huggingface.co/Xenova/clip-vit-base-patch32/tree/main/onnx"
        )
        logging.error(
            "  Text: https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1/tree/main/onnx"
        )
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
