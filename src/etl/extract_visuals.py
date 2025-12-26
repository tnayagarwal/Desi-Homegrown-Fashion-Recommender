"""
ETL Step 1 - Visual Feature Extraction
=======================================
Extracts 512-dimensional visual embeddings from fashion catalog images
using a pretrained Vision Transformer (ViT-B/32) from Hugging Face.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModel

logger = logging.getLogger(__name__)

MODEL_ID = "google/vit-base-patch32-224-in21k"
BATCH_SIZE = 128


def load_model(device: str = "cpu") -> Tuple:
    """
    Load the pretrained ViT image processor and encoder.

    Args:
        device: Target device ('cpu' or 'cuda').

    Returns:
        Tuple of (processor, model)
    """
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device)
    model.eval()
    logger.info("ViT model loaded on %s.", device)
    return processor, model


def process_image(image_path: str, processor, model, device: str = "cpu") -> List[float]:
    """
    Extract a single image's visual embedding.

    Args:
        image_path: Path to the image file.
        processor: Hugging Face image processor.
        model: Pretrained ViT encoder.
        device: Target device.

    Returns:
        A 512-dimensional embedding as a Python list.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError) as e:
        logger.error("Cannot open image '%s': %s", image_path, e)
        raise

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().tolist()
    return embedding


def batch_extract(image_paths: List[str], processor, model, device: str = "cpu") -> List[Tuple[str, List[float]]]:
    """
    Extract embeddings for a batch of images.

    Args:
        image_paths: List of image file paths.
        processor: Hugging Face image processor.
        model: Pretrained ViT encoder.
        device: Target device.

    Returns:
        List of (image_path, embedding) tuples.
    """
    results = []
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch = image_paths[i : i + BATCH_SIZE]
        logger.info("Processing batch %d/%d (%d images).", i // BATCH_SIZE + 1, -(-len(image_paths) // BATCH_SIZE), len(batch))
        for path in batch:
            try:
                emb = process_image(path, processor, model, device)
                results.append((path, emb))
            except Exception as e:
                logger.warning("Skipping %s: %s", path, e)
    return results
