#!/usr/bin/env python3
"""
Complete training script for Parametric UMAP and Vec2Text models.

This script trains both models using your existing ArXiv data and saves them
for use in the embedding-to-text pipeline.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse
import sys
import os

# Add the backend directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from app.services.embedding_to_text.parametric_umap.parametric_umap_service import ParametricUMAPService
from app.services.embedding_to_text.models.parametric_umap import ParametricUMAPConfig
from app.services.embedding_to_text.text_generation.vec2text_service import Vec2TextService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_training_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Load training data from the ArXiv dataset.
    
    Args:
        data_dir: Path to the data directory (e.g., data/arxiv_aug_sep)
        
    Returns:
        Tuple of (embeddings, coordinates, documents)
    """
    logger.info(f"Loading training data from {data_dir}")
    
    # Load embeddings
    embeddings_file = data_dir / "all_embeddings.json"
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
    
    with open(embeddings_file, 'r') as f:
        embeddings_data = json.load(f)
    
    # Load coordinates
    coordinates_file = data_dir / "coordinates.json"
    if not coordinates_file.exists():
        raise FileNotFoundError(f"Coordinates file not found: {coordinates_file}")
    
    with open(coordinates_file, 'r') as f:
        coordinates_data = json.load(f)
    
    # Load metadata
    metadata_file = data_dir / "all_metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        metadata_data = json.load(f)
    
    # Combine data
    embeddings = []
    coordinates = []
    documents = []
    
    for emb_doc in embeddings_data:
        doc_id = emb_doc['document_id']
        
        # Get coordinates
        if doc_id not in coordinates_data:
            logger.warning(f"No coordinates found for document {doc_id}")
            continue
        
        # Get metadata
        meta_doc = next((m for m in metadata_data if m['document_id'] == doc_id), None)
        if not meta_doc:
            logger.warning(f"No metadata found for document {doc_id}")
            continue
        
        embeddings.append(emb_doc['embedding'])
        coordinates.append(coordinates_data[doc_id])
        
        documents.append({
            'document_id': doc_id,
            'title': meta_doc.get('title', ''),
            'content': meta_doc.get('content', ''),
            'embedding': emb_doc['embedding'],
            'coordinates': coordinates_data[doc_id]
        })
    
    logger.info(f"Loaded {len(embeddings)} training samples")
    return np.array(embeddings), np.array(coordinates), documents


def train_parametric_umap(
    embeddings: np.ndarray, 
    coordinates: np.ndarray, 
    model_path: Path,
    config: ParametricUMAPConfig
) -> ParametricUMAPService:
    """
    Train the Parametric UMAP model.
    
    Args:
        embeddings: High-dimensional embeddings
        coordinates: 2D coordinates
        model_path: Path to save the trained model
        config: Parametric UMAP configuration
        
    Returns:
        Trained ParametricUMAPService
    """
    logger.info("Training Parametric UMAP model...")
    
    # Initialize service
    service = ParametricUMAPService(config)
    
    # Train the model
    trained_model = service.train_model(
        embeddings=embeddings,
        coordinates=coordinates,
        save_path=model_path
    )
    
    # Evaluate the model
    metrics = service.evaluate_model()
    logger.info(f"Parametric UMAP training completed. Metrics: {metrics}")
    
    return service


def train_vec2text(
    documents: List[Dict[str, Any]], 
    model_path: Path,
    epochs: int = 3,
    batch_size: int = 4
) -> Vec2TextService:
    """
    Train the Vec2Text model.
    
    Args:
        documents: Training documents with embeddings and text
        model_path: Path to save the trained model
        epochs: Number of training epochs
        batch_size: Training batch size
        
    Returns:
        Trained Vec2TextService
    """
    logger.info("Training Vec2Text model...")
    
    # Initialize service
    service = Vec2TextService(
        model_name="microsoft/DialoGPT-medium"
    )
    
    # Train the model
    training_results = service.train_on_documents(
        documents=documents,
        epochs=epochs,
        batch_size=batch_size
    )
    
    logger.info(f"Vec2Text training completed. Results: {training_results}")
    
    # Save the model
    service.save_model(model_path)
    
    return service


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Parametric UMAP and Vec2Text models")
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data/arxiv_aug_sep",
        help="Path to the training data directory"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data/models",
        help="Path to save trained models"
    )
    parser.add_argument(
        "--umap-epochs", 
        type=int, 
        default=100,
        help="Number of epochs for Parametric UMAP training"
    )
    parser.add_argument(
        "--vec2text-epochs", 
        type=int, 
        default=3,
        help="Number of epochs for Vec2Text training"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--skip-vec2text", 
        action="store_true",
        help="Skip Vec2Text training (only train Parametric UMAP)"
    )
    parser.add_argument(
        "--skip-umap", 
        action="store_true",
        help="Skip Parametric UMAP training (only train Vec2Text)"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load training data
        embeddings, coordinates, documents = load_training_data(data_dir)
        
        if len(embeddings) == 0:
            raise ValueError("No training data found")
        
        logger.info(f"Training with {len(embeddings)} samples")
        logger.info(f"Embedding dimension: {embeddings.shape[1]}")
        logger.info(f"Coordinate dimension: {coordinates.shape[1]}")
        
        # Configure Parametric UMAP
        umap_config = ParametricUMAPConfig(
            n_components=2,
            encoder_layers=(256, 128, 64),
            decoder_layers=(64, 128, 256),
            epochs=args.umap_epochs,
            batch_size=args.batch_size,
            learning_rate=0.001
        )
        
        # Train Parametric UMAP
        if not args.skip_umap:
            umap_model_path = output_dir / "parametric_umap"
            umap_service = train_parametric_umap(
                embeddings, coordinates, umap_model_path, umap_config
            )
            logger.info("Parametric UMAP training completed successfully")
        else:
            logger.info("Skipping Parametric UMAP training")
        
        # Train Vec2Text
        if not args.skip_vec2text:
            vec2text_model_path = output_dir / "vec2text"
            vec2text_service = train_vec2text(
                documents, vec2text_model_path, args.vec2text_epochs, args.batch_size
            )
            logger.info("Vec2Text training completed successfully")
        else:
            logger.info("Skipping Vec2Text training")
        
        logger.info("All training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()








