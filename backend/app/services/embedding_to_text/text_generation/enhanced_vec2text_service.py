#!/usr/bin/env python3
"""
Enhanced Vec2Text Service with improved coordinate precision.
This service uses a more sophisticated approach to generate text that maps precisely to target coordinates.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import json
import pickle
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, GPT2LMHeadModel

logger = logging.getLogger(__name__)

class EnhancedVec2TextNetwork(nn.Module):
    """Enhanced Vec2Text network with improved architecture."""
    
    def __init__(self, embedding_dim: int = 384, coordinate_dim: int = 2, 
                 hidden_dim: int = 768, vocab_size: int = 50257, prefix_length: int = 10):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.coordinate_dim = coordinate_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.prefix_length = prefix_length
        
        # Input projection layers
        self.embedding_proj = nn.Linear(embedding_dim, hidden_dim)
        self.coordinate_proj = nn.Linear(coordinate_dim, hidden_dim)
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Transformer encoder for generating prefix embeddings
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Output projection to vocab size
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        # Coordinate consistency head
        self.coordinate_head = nn.Linear(hidden_dim, coordinate_dim)
        
    def forward(self, embeddings: torch.Tensor, coordinates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both token predictions and coordinate predictions."""
        batch_size = embeddings.size(0)
        
        # Project inputs
        emb_proj = self.embedding_proj(embeddings)  # [batch, hidden_dim]
        coord_proj = self.coordinate_proj(coordinates)  # [batch, hidden_dim]
        
        # Fuse information
        fused = torch.cat([emb_proj, coord_proj], dim=-1)  # [batch, hidden_dim * 2]
        fused = self.fusion(fused)  # [batch, hidden_dim]
        
        # Create sequence for transformer (repeat fused vector for prefix_length)
        sequence = fused.unsqueeze(1).repeat(1, self.prefix_length, 1)  # [batch, prefix_length, hidden_dim]
        
        # Apply transformer
        transformer_output = self.transformer(sequence)  # [batch, prefix_length, hidden_dim]
        
        # Generate token predictions
        token_predictions = self.output_proj(transformer_output)  # [batch, prefix_length, vocab_size]
        
        # Generate coordinate predictions (average over sequence)
        avg_output = torch.mean(transformer_output, dim=1)  # [batch, hidden_dim]
        coordinate_predictions = self.coordinate_head(avg_output)  # [batch, coordinate_dim]
        
        return token_predictions, coordinate_predictions
    
    def get_prefix_embeddings(self, embeddings: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """Get the prefix embeddings for text generation."""
        batch_size = embeddings.size(0)
        
        # Project inputs
        emb_proj = self.embedding_proj(embeddings)
        coord_proj = self.coordinate_proj(coordinates)
        
        # Fuse information
        fused = torch.cat([emb_proj, coord_proj], dim=-1)
        fused = self.fusion(fused)
        
        # Create sequence for transformer
        sequence = fused.unsqueeze(1).repeat(1, self.prefix_length, 1)
        
        # Apply transformer
        transformer_output = self.transformer(sequence)
        
        return transformer_output  # [batch, prefix_length, hidden_dim]

class EnhancedVec2TextService:
    """Enhanced Vec2Text service with improved precision."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
        
        # Initialize Vec2Text network
        self.vec2text_network = None
        self.optimizer = None
        self.prefix_length = 10
        
        # Load trained Parametric UMAP
        self.parametric_umap = None
        self._load_parametric_umap()
        
        # Training data
        self.training_embeddings = None
        self.training_texts = None
        self.training_coordinates = None
        
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
    
    def load_training_data(self, max_samples: int = 1000):
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
            else:
                logger.error("Cannot generate coordinates without Parametric UMAP")
                return False
            
            logger.info(f"Loaded {len(embeddings)} training samples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return False
    
    def train_network(self, epochs: int = 100, batch_size: int = 8, learning_rate: float = 1e-4):
        """Train the Vec2Text network."""
        if self.training_embeddings is None:
            logger.error("No training data loaded")
            return False
        
        logger.info("Initializing Vec2Text network...")
        
        # Initialize network
        self.vec2text_network = EnhancedVec2TextNetwork(
            embedding_dim=384,
            coordinate_dim=2,
            hidden_dim=768,
            vocab_size=self.tokenizer.vocab_size
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.vec2text_network.parameters(), lr=learning_rate)
        
        # Convert data to tensors
        embeddings_tensor = torch.tensor(self.training_embeddings, dtype=torch.float32).to(self.device)
        coordinates_tensor = torch.tensor(self.training_coordinates, dtype=torch.float32).to(self.device)
        
        # Tokenize texts
        tokenized_texts = []
        for text in self.training_texts:
            tokens = self.tokenizer.encode(text, max_length=100, truncation=True, padding='max_length')
            tokenized_texts.append(tokens)
        
        texts_tensor = torch.tensor(tokenized_texts, dtype=torch.long).to(self.device)
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        # Training loop
        for epoch in range(epochs):
            self.vec2text_network.train()
            
            # Create batches
            num_batches = len(embeddings_tensor) // batch_size
            total_loss = 0
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                batch_embeddings = embeddings_tensor[start_idx:end_idx]
                batch_coordinates = coordinates_tensor[start_idx:end_idx]
                batch_texts = texts_tensor[start_idx:end_idx]
                
                # Forward pass
                self.optimizer.zero_grad()
                
                token_predictions, coordinate_predictions = self.vec2text_network(
                    batch_embeddings, batch_coordinates
                )
                
                # Calculate losses
                # Token prediction loss - only use the first prefix_length tokens
                target_tokens = batch_texts[:, :self.prefix_length]  # [batch_size, prefix_length]
                token_loss = nn.CrossEntropyLoss()(
                    token_predictions.reshape(-1, self.tokenizer.vocab_size),
                    target_tokens.reshape(-1)
                )
                
                # Coordinate consistency loss
                coord_loss = nn.MSELoss()(coordinate_predictions, batch_coordinates)
                
                # Total loss
                total_batch_loss = token_loss + 0.1 * coord_loss
                
                # Backward pass
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vec2text_network.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += total_batch_loss.item()
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        logger.info("Training completed")
        return True
    
    def generate_text_for_coordinates(self, coordinates: List[float], max_length: int = 150) -> str:
        """Generate text for given coordinates."""
        if self.vec2text_network is None:
            logger.error("Vec2Text network not trained")
            return "Network not trained"
        
        if self.parametric_umap is None:
            logger.error("Parametric UMAP not available")
            return "UMAP not available"
        
        try:
            # Get embedding from coordinates using inverse UMAP
            target_embedding = self.parametric_umap.get_target_embedding(coordinates)
            if target_embedding is None:
                logger.error("Failed to get target embedding")
                return "Failed to get embedding"
            
            # Convert to tensor
            embedding_tensor = torch.tensor([target_embedding], dtype=torch.float32).to(self.device)
            coord_tensor = torch.tensor([coordinates], dtype=torch.float32).to(self.device)
            
            # Get prefix embeddings
            self.vec2text_network.eval()
            with torch.no_grad():
                prefix_embeddings = self.vec2text_network.get_prefix_embeddings(
                    embedding_tensor, coord_tensor
                )  # [1, prefix_length, hidden_dim]
                
                # Generate text using GPT-2 with prefix embeddings
                # Convert prefix embeddings to tokens (simplified approach)
                prefix_tokens = torch.argmax(
                    self.vec2text_network.output_proj(prefix_embeddings), dim=-1
                )  # [1, prefix_length]
                
                # Generate continuation
                generated_tokens = self.gpt2_model.generate(
                    input_ids=prefix_tokens,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode to text
                generated_text = self.tokenizer.decode(
                    generated_tokens[0], skip_special_tokens=True
                )
                
                return generated_text.strip()
                
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error: {str(e)}"
    
    def validate_precision(self, num_samples: int = 100) -> Dict[str, float]:
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
    
    def save_model(self, model_path: Path):
        """Save the trained model."""
        if self.vec2text_network is None:
            logger.error("No trained model to save")
            return False
        
        try:
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Save network state
            torch.save(self.vec2text_network.state_dict(), model_path / "vec2text_network.pth")
            
            # Save configuration
            config = {
                "embedding_dim": 384,
                "coordinate_dim": 2,
                "hidden_dim": 768,
                "vocab_size": self.tokenizer.vocab_size,
                "prefix_length": 10
            }
            
            with open(model_path / "config.json", 'w') as f:
                json.dump(config, f)
            
            logger.info(f"Model saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, model_path: Path):
        """Load a trained model."""
        try:
            if not model_path.exists():
                logger.error(f"Model path does not exist: {model_path}")
                return False
            
            # Load configuration
            config_file = model_path / "config.json"
            if not config_file.exists():
                logger.error("Configuration file not found")
                return False
            
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Initialize network
            self.vec2text_network = EnhancedVec2TextNetwork(**config).to(self.device)
            
            # Load weights
            weights_file = model_path / "vec2text_network.pth"
            if not weights_file.exists():
                logger.error("Weights file not found")
                return False
            
            self.vec2text_network.load_state_dict(torch.load(weights_file, map_location=self.device))
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

def main():
    """Main function for training and testing."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize service
    service = EnhancedVec2TextService()
    
    # Load training data
    if not service.load_training_data(max_samples=1000):
        logger.error("Failed to load training data")
        return
    
    # Train network
    if not service.train_network(epochs=100):
        logger.error("Failed to train network")
        return
    
    # Validate precision
    results = service.validate_precision(num_samples=50)
    logger.info(f"Final validation results: {results}")
    
    # Save model
    model_path = Path(__file__).parent.parent.parent.parent.parent / "data" / "models" / "enhanced_vec2text"
    if service.save_model(model_path):
        logger.info("Model saved successfully")
    
    # Test generation
    test_coords = [0.0, 0.0]
    generated_text = service.generate_text_for_coordinates(test_coords)
    logger.info(f"Test generation for {test_coords}: {generated_text}")

if __name__ == "__main__":
    main()
