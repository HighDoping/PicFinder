# -*- coding: utf-8 -*-
"""
Multilingual CLIP model inference using ONNXRuntime for visual similarity search.

Uses the official sentence-transformers multilingual CLIP approach:
- Vision: Original clip-ViT-B-32 for encoding images
- Text: clip-ViT-B-32-multilingual-v1 for encoding text (supports 50+ languages)

The multilingual text encoder maps text from multiple languages into the same
embedding space as the vision encoder, enabling cross-lingual image-text search.
"""

import logging
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

is_nuitka = "__compiled__" in globals()

if is_nuitka or getattr(sys, "frozen", False):
    models_dir = Path(sys.argv[0]).parent / "models"
else:
    models_dir = Path(__file__).resolve().parent.parent / "models"


class CLIPModel:
    """
    Multilingual CLIP model for encoding images and text into embeddings.

    Uses the official sentence-transformers approach:
    - Vision: Original clip-ViT-B-32 for encoding images
    - Text: clip-ViT-B-32-multilingual-v1 aligned to vision model (50+ languages)

    The multilingual text encoder maps text from multiple languages into the same
    embedding space as the vision encoder, enabling cross-lingual image-text search.
    """

    def __init__(self, model_name: str = "clip-vit-b-32-multilingual"):
        """
        Initialize multilingual CLIP model with separate vision and text encoders.

        Args:
            model_name: Base name for the model files (default: "clip-vit-b-32-multilingual")
        """
        self.model_name = model_name
        self.vision_model_path = models_dir / f"{model_name}-vision.onnx"
        self.text_model_path = models_dir / f"{model_name}-text.onnx"
        self.tokenizer_path = models_dir / "clip-tokenizer.json"

        # Check if models exist
        if not self.vision_model_path.exists():
            raise FileNotFoundError(
                f"Vision model not found: {self.vision_model_path}\n"
                f"Please run: python src/dev/download_clip.py"
            )

        if not self.text_model_path.exists():
            raise FileNotFoundError(
                f"Text model not found: {self.text_model_path}\n"
                f"Please run: python src/dev/download_clip.py"
            )

        # Initialize ONNX Runtime sessions
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        try:
            self.vision_session = ort.InferenceSession(
                str(self.vision_model_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            self.text_session = ort.InferenceSession(
                str(self.text_model_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            logging.info(f"Loaded multilingual CLIP model: {model_name}")
            logging.info("  Vision: Original CLIP ViT-B/32")
            logging.info("  Text: Multilingual encoder (50+ languages)")

            # Log input/output info for debugging
            logging.debug("Vision model inputs:")
            for inp in self.vision_session.get_inputs():
                logging.debug(f"  {inp.name}: {inp.shape} ({inp.type})")
            logging.debug("Text model inputs:")
            for inp in self.text_session.get_inputs():
                logging.debug(f"  {inp.name}: {inp.shape} ({inp.type})")
            logging.debug("Text model outputs:")
            for out in self.text_session.get_outputs():
                logging.debug(f"  {out.name}: {out.shape} ({out.type})")

        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP ONNX models: {e}")

        # Initialize tokenizer
        tokenizer_json_path = models_dir / "clip-tokenizer.json"

        if not tokenizer_json_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found: {tokenizer_json_path}\n"
                f"Please run: python src/dev/download_clip.py"
            )

        self.tokenizer = Tokenizer.from_file(str(tokenizer_json_path))
        logging.info("Loaded HuggingFace tokenizer for multilingual support")

        # CLIP preprocessing constants
        self.image_size = 224
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        self.std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for CLIP vision encoder.

        Args:
            image: Input image in BGR format (OpenCV default)

        Returns:
            Preprocessed image tensor [1, 3, 224, 224]
        """
        # Convert BGR to RGB
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:  # BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to 224x224
        image = cv2.resize(
            image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Standardize with CLIP mean/std
        image = (image - self.mean) / self.std

        # Transpose to CHW format and add batch dimension
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        return image

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """
        Encode image into embedding vector using original CLIP vision encoder.

        Args:
            image: Input image in BGR format (OpenCV default)

        Returns:
            Normalized embedding vector [512]
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image)

        # Run inference with original CLIP ViT-B/32 vision encoder
        input_name = self.vision_session.get_inputs()[0].name
        output_name = self.vision_session.get_outputs()[0].name

        embedding = self.vision_session.run([output_name], {input_name: image_tensor})[
            0
        ]

        # Normalize embedding (take pooler output if available)
        if len(embedding.shape) > 2:
            # If output is [batch, seq_len, hidden_dim], take the first token (CLS)
            embedding = embedding[:, 0, :]

        embedding = embedding.flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text into embedding vector using multilingual text encoder.

        The sentence-transformers multilingual model is aligned to the
        original CLIP vision encoder and supports 50+ languages.

        Args:
            text: Input text string in any supported language

        Returns:
            Normalized embedding vector [512]
        """
        # Tokenize text using HuggingFace tokenizer
        encoding = self.tokenizer.encode(text)
        token_inputs = {
            "input_ids": np.array([encoding.ids], dtype=np.int64),
            "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
        }

        # Run inference
        input_names = [inp.name for inp in self.text_session.get_inputs()]
        outputs = self.text_session.get_outputs()

        feed_dict = {}
        if "input_ids" in input_names:
            feed_dict["input_ids"] = token_inputs["input_ids"]
        if "attention_mask" in input_names:
            feed_dict["attention_mask"] = token_inputs["attention_mask"]

        # For models that might need token_type_ids (BERT-based models)
        if "token_type_ids" in input_names:
            batch_size = token_inputs["input_ids"].shape[0]
            seq_len = token_inputs["input_ids"].shape[1]
            feed_dict["token_type_ids"] = np.zeros(
                (batch_size, seq_len), dtype=np.int64
            )

        # Run the model
        output_names = [out.name for out in outputs]
        results = self.text_session.run(output_names, feed_dict)

        # Sentence-transformers ONNX models may have multiple outputs
        # The first output is usually "last_hidden_state" (sequence output)
        # If there's a "sentence_embedding" output, use that instead
        embedding = None
        for idx, out_name in enumerate(output_names):
            if (
                "sentence_embedding" in out_name.lower()
                or "pooler_output" in out_name.lower()
            ):
                embedding = results[idx]
                break

        # If no pooled output found, use the first output and pool it
        if embedding is None:
            embedding = results[0]

            # If output is [batch, seq_len, hidden_dim], apply mean pooling
            if len(embedding.shape) > 2:
                attention_mask_expanded = token_inputs["attention_mask"][
                    :, :, np.newaxis
                ].astype(np.float32)
                sum_embeddings = np.sum(embedding * attention_mask_expanded, axis=1)
                sum_mask = np.clip(
                    attention_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None
                )
                embedding = sum_embeddings / sum_mask

        embedding = embedding.flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding


def test_clip_model():
    """Test multilingual CLIP model loading and inference."""
    import time

    print("Testing multilingual CLIP model...")

    try:
        model = CLIPModel()
        print(f"✓ Model loaded successfully")

        # Test text encoding in multiple languages
        test_texts = [
            ("a photo of a cat", "English"),
            ("一只猫的照片", "Chinese"),
            ("une photo d'un chat", "French"),
            ("Foto von einer Katze", "German"),
        ]

        embeddings = []
        for text, lang in test_texts:
            t0 = time.perf_counter()
            text_emb = model.encode_text(text)
            t1 = time.perf_counter()
            embeddings.append(text_emb)
            print(
                f"✓ Text encoding ({lang}): {text_emb.shape}, took {(t1-t0)*1000:.2f}ms"
            )
            print(f"  Input: {text}")
            print(f"  Sample values: {text_emb[:5]}")

        # Test image encoding (dummy image)
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        image_emb = model.encode_image(dummy_image)
        t1 = time.perf_counter()
        print(f"\n✓ Image encoding: {image_emb.shape}, took {(t1-t0)*1000:.2f}ms")
        print(f"  Sample values: {image_emb[:5]}")

        # Test cross-lingual similarity
        print("\nCross-lingual text-image similarities:")
        for (text, lang), text_emb in zip(test_texts, embeddings):
            similarity = np.dot(text_emb, image_emb)
            print(f"  {lang}: {similarity:.4f}")

        # Test that similar texts in different languages produce similar embeddings
        print("\nCross-lingual text similarities (should be high for same meaning):")
        for i, (text1, lang1) in enumerate(test_texts):
            for j, (text2, lang2) in enumerate(test_texts):
                if i < j:
                    similarity = np.dot(embeddings[i], embeddings[j])
                    print(f"  {lang1} <-> {lang2}: {similarity:.4f}")

        print("\n✓ All tests passed!")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(test_clip_model())
