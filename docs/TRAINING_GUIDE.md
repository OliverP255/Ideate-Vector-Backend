# Training Guide: Vec2Text and Parametric UMAP

This guide explains how to train both the Parametric UMAP and Vec2Text models for the embedding-to-text pipeline.

## Overview

The embedding-to-text pipeline consists of two main components:

1. **Parametric UMAP**: Provides invertible dimensionality reduction (2D ↔ embedding)
2. **Vec2Text**: Generates text directly from embeddings with iterative correction

## Data Requirements

### Required Files

Your training data should be organized as follows:

```
data/arxiv_aug_sep/
├── all_embeddings.json      # Document embeddings
├── all_metadata.json        # Document metadata (title, content)
└── coordinates.json         # 2D coordinates from existing UMAP
```

### Data Format

#### `all_embeddings.json`
```json
[
  {
    "document_id": "arxiv_2409.05988v3",
    "embedding": [0.1, 0.2, 0.3, ...]  // High-dimensional vector
  }
]
```

#### `all_metadata.json`
```json
[
  {
    "document_id": "arxiv_2409.05988v3",
    "title": "Transmon qubit modeling and characterization...",
    "content": "This study presents the design...",
    "authors": ["R. Moretti", "D. Labranca", ...]
  }
]
```

#### `coordinates.json`
```json
{
  "arxiv_2409.05988v3": [0.5, -0.3],  // [x, y] coordinates
  "arxiv_2409.05989v1": [0.2, 0.8]
}
```

## Training Process

### 1. Parametric UMAP Training

**Purpose**: Learn bidirectional mapping between embeddings and 2D coordinates.

**Training Steps**:
1. **Autoencoder Training**: Joint training of encoder+decoder for reconstruction
2. **Encoder Fine-tuning**: Fine-tune encoder to match existing 2D coordinates
3. **Model Evaluation**: Calculate reconstruction accuracy metrics

**Key Parameters**:
- `encoder_layers`: (256, 128, 64) - Neural network architecture for embedding → 2D
- `decoder_layers`: (64, 128, 256) - Neural network architecture for 2D → embedding
- `epochs`: 100 - Training iterations
- `batch_size`: 32 - Batch size for training

### 2. Vec2Text Training

**Purpose**: Generate text directly from embeddings with iterative correction.

**Training Steps**:
1. **Model Initialization**: Load base language model (DialoGPT-medium)
2. **Document Processing**: Prepare (embedding, text) pairs
3. **Fine-tuning**: Train model to generate text from embeddings
4. **Iterative Correction**: Implement correction loop for precise placement

**Key Parameters**:
- `max_length`: 300 - Maximum generated text length
- `num_beams`: 4 - Beam search width
- `temperature`: 0.7 - Sampling temperature
- `max_correction_iterations`: 5 - Iterative correction rounds
- `embedding_tolerance`: 0.1 - Convergence threshold

## Running the Training

### Quick Start

```bash
# Train both models with default settings
python scripts/train_models.py

# Train with custom data directory
python scripts/train_models.py --data-dir data/arxiv_test

# Train only Parametric UMAP
python scripts/train_models.py --skip-vec2text

# Train only Vec2Text
python scripts/train_models.py --skip-umap

# Custom training parameters
python scripts/train_models.py \
  --umap-epochs 150 \
  --vec2text-epochs 5 \
  --batch-size 16
```

### Command Line Options

- `--data-dir`: Path to training data directory (default: `data/arxiv_aug_sep`)
- `--output-dir`: Path to save trained models (default: `data/models`)
- `--umap-epochs`: Number of epochs for Parametric UMAP (default: 100)
- `--vec2text-epochs`: Number of epochs for Vec2Text (default: 3)
- `--batch-size`: Batch size for training (default: 32)
- `--skip-vec2text`: Skip Vec2Text training
- `--skip-umap`: Skip Parametric UMAP training

## Training Output

### Model Files

After training, models are saved to:

```
data/models/
├── parametric_umap/
│   ├── config.pkl           # Model configuration
│   ├── scaler.pkl           # Coordinate scaler
│   ├── encoder/             # Encoder model (embedding → 2D)
│   └── decoder/             # Decoder model (2D → embedding)
└── vec2text/
    ├── vec2text_config.json # Vec2Text configuration
    ├── tokenizer/           # Text tokenizer
    ├── language_model/      # Fine-tuned language model
    └── embedding_model/     # Embedding model
```

### Training Metrics

#### Parametric UMAP Metrics
- `coordinate_mse`: Mean squared error for 2D coordinate prediction
- `coordinate_mae`: Mean absolute error for 2D coordinate prediction
- `embedding_mse`: Mean squared error for embedding reconstruction
- `embedding_mae`: Mean absolute error for embedding reconstruction
- `avg_cosine_similarity`: Average cosine similarity between original and reconstructed embeddings

#### Vec2Text Metrics
- `average_embedding_distance`: Average distance between target and generated embeddings
- `average_iterations`: Average number of correction iterations needed
- `convergence_rate`: Percentage of generations that converged within tolerance

## Hardware Requirements

### Parametric UMAP
- **CPU**: Multi-core recommended for faster training
- **RAM**: 8GB+ recommended for large datasets
- **Storage**: 1-2GB for model files

### Vec2Text
- **GPU**: CUDA-compatible GPU recommended (8GB+ VRAM)
- **RAM**: 16GB+ recommended
- **Storage**: 5-10GB for model files

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use smaller datasets
2. **Slow Training**: Use GPU for Vec2Text, increase batch size if memory allows
3. **Poor Convergence**: Increase epochs, adjust learning rate, check data quality
4. **Missing Dependencies**: Install required packages (TensorFlow, PyTorch, transformers)

### Performance Tips

1. **Start Small**: Use `data/arxiv_test` (10 documents) for initial testing
2. **Monitor Metrics**: Watch training metrics to detect overfitting
3. **Save Checkpoints**: Models are automatically saved during training
4. **Validate Data**: Ensure all documents have both embeddings and coordinates

## Using Trained Models

After training, models can be loaded and used:

```python
from backend.app.services.embedding_to_text.embedding_to_text_service import EmbeddingToTextService

# Initialize service with trained models
service = EmbeddingToTextService()
service.load_trained_model(Path("data/models"))

# Generate text from coordinates
from backend.app.services.embedding_to_text.models.base import EmbeddingToTextRequest

request = EmbeddingToTextRequest(
    x=0.5,  # X coordinate on map
    y=-0.3, # Y coordinate on map
    method="vec2text"  # or "llm_synthesis"
)

response = service.generate_text_from_coordinates(request)
print(f"Generated text: {response.text_generation_result.title}")
```

## Next Steps

1. **Evaluate Quality**: Test generated text quality and semantic placement
2. **Fine-tune Parameters**: Adjust model parameters based on results
3. **Scale Up**: Train on larger datasets for better performance
4. **Integration**: Integrate trained models into your application pipeline








