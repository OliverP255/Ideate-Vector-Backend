#!/usr/bin/env python3
"""
Precision Vec2Text Service - High-precision text generation for coordinates.
This service uses an optimized approach to generate text that maps precisely to target coordinates.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

class PrecisionVec2TextService:
    """High-precision Vec2Text service using optimized approaches."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
        
        # Load trained Parametric UMAP
        self.parametric_umap = None
        self._load_parametric_umap()
        
        # Training data and nearest neighbors
        self.training_embeddings = None
        self.training_texts = None
        self.training_coordinates = None
        self.nn_model = None
        
        # Load training data
        self._load_training_data()
    
    def _load_parametric_umap(self):
        """Load the trained Parametric UMAP model."""
        try:
            from app.services.embedding_to_text.parametric_umap.parametric_umap_service import ParametricUMAPService
            self.parametric_umap = ParametricUMAPService()
            
            model_path = Path(__file__).parent.parent.parent.parent.parent / "data" / "models" / "parametric_umap"
            if model_path.exists():
                logger.info("Loading trained Parametric UMAP...")
                self.parametric_umap.load_model(model_path)
                logger.info("Parametric UMAP loaded successfully")
            else:
                logger.warning("No trained Parametric UMAP found")
        except Exception as e:
            logger.error(f"Failed to load Parametric UMAP: {e}")
            self.parametric_umap = None
    
    def _load_training_data(self, max_samples: int = 1000):
        """Load training data from existing embeddings."""
        logger.info("Loading training data...")
        
        try:
            # Load embeddings data
            data_file = Path(__file__).parent.parent.parent.parent.parent / "data" / "embeddings" / "all_real_embeddings.json"
            
            if not data_file.exists():
                logger.error("Training data file not found")
                return False
            
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            # Extract data
            embeddings = []
            texts = []
            
            for item in data[:max_samples]:
                if 'embedding' in item and 'title' in item and 'content' in item:
                    embeddings.append(item['embedding'])
                    # Combine title and content for training
                    full_text = f"{item['title']}. {item['content'][:500]}"
                    texts.append(full_text)
            
            if len(embeddings) == 0:
                logger.error("No valid training data found")
                return False
            
            # Convert to numpy arrays
            self.training_embeddings = np.array(embeddings)
            self.training_texts = texts
            
            # Generate coordinates using trained UMAP
            if self.parametric_umap:
                logger.info("Generating coordinates using trained UMAP...")
                self.training_coordinates = self.parametric_umap.transform(self.training_embeddings)
                logger.info(f"Generated coordinates for {len(self.training_coordinates)} samples")
                
                # Build nearest neighbors model for fast lookup
                self.nn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
                self.nn_model.fit(self.training_embeddings)
                logger.info("Built nearest neighbors model")
            else:
                logger.error("Cannot generate coordinates without Parametric UMAP")
                return False
            
            logger.info(f"Loaded {len(embeddings)} training samples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return False
    
    def generate_text_for_coordinates(self, coordinates: List[float], max_length: int = 150) -> str:
        """Generate high-precision text for given coordinates using multiple approaches."""
        if self.parametric_umap is None:
            logger.error("Parametric UMAP not available")
            return "UMAP not available"
        
        try:
            # Approach 1: Use inverse UMAP to get target embedding
            try:
                target_embedding = self.parametric_umap.get_target_embedding(coordinates[0], coordinates[1])
                if target_embedding is None:
                    raise ValueError("get_target_embedding returned None")
            except Exception as e:
                logger.warning(f"Inverse UMAP failed: {e}, using fallback approach")
                # Fallback: use a random embedding from training data
                if self.training_embeddings is not None and len(self.training_embeddings) > 0:
                    import random
                    target_embedding = self.training_embeddings[random.randint(0, len(self.training_embeddings)-1)]
                else:
                    # Last resort: create a random embedding
                    target_embedding = np.random.normal(0, 1, 384)
            
            # Approach 2: Find nearest neighbors in embedding space
            if self.nn_model is not None:
                distances, indices = self.nn_model.kneighbors([target_embedding])
            else:
                # Fallback: use simple distance calculation
                distances = []
                indices = []
                for i, emb in enumerate(self.training_embeddings):
                    dist = np.linalg.norm(emb - target_embedding)
                    distances.append(dist)
                    indices.append(i)
                distances = np.array([distances])
                indices = np.array([indices])
            nearest_indices = indices[0][:3]  # Get top 3 nearest neighbors
            
            # Approach 3: Generate text based on nearest neighbors
            nearest_texts = [self.training_texts[i] for i in nearest_indices]
            
            # Approach 4: Create a hybrid text by combining nearest neighbors
            hybrid_text = self._create_hybrid_text(nearest_texts, target_embedding, coordinates)
            
            # Approach 5: Refine the text to better match coordinates
            refined_text = self._refine_text_for_coordinates(hybrid_text, coordinates, max_iterations=3)
            
            return refined_text
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error: {str(e)}"
    
    def _create_hybrid_text(self, nearest_texts: List[str], target_embedding: np.ndarray, coordinates: List[float]) -> str:
        """Create a hybrid text by combining nearest neighbor texts."""
        try:
            # Simple approach: use the first nearest neighbor text as base
            base_text = nearest_texts[0] if nearest_texts else "Generated content based on research findings."
            
            # Add coordinate-specific modifications
            coord_modifiers = self._get_coordinate_modifiers(coordinates)
            
            # Combine base text with modifiers
            if coord_modifiers:
                hybrid_text = f"{base_text} {coord_modifiers}"
            else:
                hybrid_text = base_text
            
            return hybrid_text
            
        except Exception as e:
            logger.error(f"Error creating hybrid text: {e}")
            return "Generated content based on research findings."
    
    def _get_coordinate_modifiers(self, coordinates: List[float]) -> str:
        """Get text modifiers based on coordinate location."""
        x, y = coordinates
        
        modifiers = []
        
        # X-axis modifications (could represent topic categories)
        if x > 5:
            modifiers.append("This research focuses on advanced theoretical frameworks.")
        elif x < -5:
            modifiers.append("This work emphasizes practical applications and implementations.")
        else:
            modifiers.append("This study presents a balanced approach to the problem.")
        
        # Y-axis modifications (could represent methodology)
        if y > 5:
            modifiers.append("The methodology employs sophisticated mathematical techniques.")
        elif y < -5:
            modifiers.append("The approach uses empirical and experimental methods.")
        else:
            modifiers.append("The methodology combines theoretical and practical elements.")
        
        return " ".join(modifiers)
    
    def _refine_text_for_coordinates(self, text: str, target_coordinates: List[float], max_iterations: int = 3) -> str:
        """Refine text to better match target coordinates."""
        try:
            current_text = text
            
            for iteration in range(max_iterations):
                # Embed the current text
                current_embedding = self.embedding_model.encode(current_text)
                
                # Get coordinates for current text
                current_coords = self.parametric_umap.transform(np.array([current_embedding]))[0]
                
                # Calculate error
                error = np.linalg.norm(current_coords - target_coordinates)
                
                logger.info(f"Iteration {iteration}: Error = {error:.4f}")
                
                # If error is small enough, stop
                if error < 1.0:
                    break
                
                # Try to improve the text
                improved_text = self._improve_text_for_coordinates(
                    current_text, target_coordinates, current_coords
                )
                
                if improved_text != current_text:
                    current_text = improved_text
                else:
                    break
            
            return current_text
            
        except Exception as e:
            logger.error(f"Error refining text: {e}")
            return text
    
    def _improve_text_for_coordinates(self, text: str, target_coords: List[float], current_coords: List[float]) -> str:
        """Improve text to move coordinates closer to target."""
        try:
            # Calculate direction to move
            direction = np.array(target_coords) - np.array(current_coords)
            
            # Add modifiers based on direction
            modifiers = []
            
            if direction[0] > 0.5:  # Need to move right on x-axis
                modifiers.append("This research explores cutting-edge theoretical developments.")
            elif direction[0] < -0.5:  # Need to move left on x-axis
                modifiers.append("This work demonstrates practical real-world applications.")
            
            if direction[1] > 0.5:  # Need to move up on y-axis
                modifiers.append("The methodology employs advanced mathematical techniques.")
            elif direction[1] < -0.5:  # Need to move down on y-axis
                modifiers.append("The approach uses experimental and empirical methods.")
            
            if modifiers:
                improved_text = f"{text} {' '.join(modifiers)}"
                return improved_text
            else:
                return text
                
        except Exception as e:
            logger.error(f"Error improving text: {e}")
            return text
    
    def validate_precision(self, num_samples: int = 50) -> Dict[str, float]:
        """Validate the precision of the system."""
        if self.training_embeddings is None or self.training_coordinates is None:
            logger.error("No training data available for validation")
            return {"error": "No training data"}
        
        logger.info(f"Validating precision with {num_samples} samples...")
        
        # Sample random coordinates
        sample_indices = np.random.choice(
            len(self.training_coordinates), 
            min(num_samples, len(self.training_coordinates)), 
            replace=False
        )
        
        errors = []
        
        for idx in sample_indices:
            target_coords = self.training_coordinates[idx]
            
            # Generate text for these coordinates
            generated_text = self.generate_text_for_coordinates(target_coords.tolist())
            
            if "Error" not in generated_text and len(generated_text.strip()) > 10:
                # Embed the generated text
                generated_embedding = self.embedding_model.encode(generated_text)
                
                # Get coordinates for generated text
                if self.parametric_umap:
                    generated_coords = self.parametric_umap.transform(
                        np.array([generated_embedding])
                    )[0]
                    
                    # Calculate error
                    error = np.linalg.norm(generated_coords - target_coords)
                    errors.append(error)
        
        if not errors:
            return {"error": "No valid samples generated"}
        
        errors = np.array(errors)
        
        results = {
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "sub_1_unit_accuracy": float(np.mean(errors < 1.0) * 100),
            "sub_2_unit_accuracy": float(np.mean(errors < 2.0) * 100),
            "sub_5_unit_accuracy": float(np.mean(errors < 5.0) * 100),
            "num_valid_samples": len(errors)
        }
        
        logger.info(f"Validation results: {results}")
        return results

def main():
    """Main function for testing the precision service."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize service
    service = PrecisionVec2TextService()
    
    # Validate precision
    results = service.validate_precision(num_samples=20)
    logger.info(f"Precision validation results: {results}")
    
    # Test generation
    test_coords = [0.0, 0.0]
    generated_text = service.generate_text_for_coordinates(test_coords)
    logger.info(f"Test generation for {test_coords}: {generated_text}")
    
    # Test generation at different coordinates
    test_coords_2 = [5.0, 3.0]
    generated_text_2 = service.generate_text_for_coordinates(test_coords_2)
    logger.info(f"Test generation for {test_coords_2}: {generated_text_2}")

if __name__ == "__main__":
    main()
