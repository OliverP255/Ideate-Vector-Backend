# Training Guide for Vec2Text and Parametric UMAP

This guide explains how to train and optimize both Vec2Text and Parametric UMAP models for the advanced gap filling system.

## Overview

The advanced gap filling system uses two key models:

1. **Parametric UMAP**: Maps high-dimensional embeddings to 2D coordinates and vice versa
2. **Vec2Text**: Generates text directly from embedding vectors

## Parametric UMAP Training

### Purpose
Parametric UMAP provides invertible dimensionality reduction, allowing us to:
- Map embeddings → 2D coordinates (for visualization)
- Map 2D coordinates → embeddings (for gap filling)

### Training Process

#### 1. Data Requirements
- **Input**: High-dimensional embeddings (typically 384-1536 dimensions)
- **Quantity**: Minimum 1000 samples, optimal 5000+ samples
- **Quality**: Well-distributed embeddings across the semantic space

#### 2. Training Configuration
```python
config = ParametricUMAPConfig(
    n_components=2,           # 2D output
    n_neighbors=15,           # Local neighborhood size
    min_dist=0.1,             # Minimum distance between points
    spread=1.0,               # Effective scale of embedded points
    n_epochs=200,             # Training epochs
    batch_size=256,           # Batch size for training
    learning_rate=0.001,      # Learning rate
    hidden_layers=[256, 128], # Neural network architecture
    validation_split=0.1      # Validation data split
)
```

#### 3. Training Steps
1. **Data Preparation**: Load embeddings and normalize
2. **Model Training**: Train neural network to approximate UMAP embedding
3. **Validation**: Test reconstruction quality
4. **Saving**: Save model for inference

#### 4. Quality Metrics
- **Reconstruction Error**: How well 2D→embedding→2D preserves original coordinates
- **Embedding Distance**: How well embedding→2D→embedding preserves semantic relationships
- **Target Accuracy**: How close generated text lands to target coordinates

### Training Command
```bash
# Initialize and train Parametric UMAP
curl -X POST "http://localhost:8000/api/advanced-gap-filling/initialize"
curl -X POST "http://localhost:8000/api/advanced-gap-filling/train-parametric-umap"
```

## Vec2Text Training

### Purpose
Vec2Text generates text directly from embedding vectors, providing:
- Direct embedding-to-text generation
- Iterative correction to match target embeddings
- Higher quality than retrieval-based methods

### Training Process

#### 1. Data Requirements
- **Input**: Document embeddings + corresponding text
- **Format**: List of documents with `embedding` and `content` fields
- **Quantity**: Minimum 500 samples, optimal 2000+ samples
- **Quality**: Diverse text samples with good embeddings

#### 2. Training Configuration
```python
vec2text_config = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "max_length": 512,
    "learning_rate": 2e-5,
    "batch_size": 8,
    "epochs": 3,
    "warmup_steps": 100,
    "weight_decay": 0.01
}
```

#### 3. Training Steps
1. **Data Preparation**: Load document-embedding pairs
2. **Model Training**: Train embedding-to-text generation
3. **Validation**: Test generation quality
4. **Saving**: Save trained model

#### 4. Quality Metrics
- **Embedding Similarity**: How similar generated text embedding is to target
- **Text Quality**: Coherence, relevance, and length of generated text
- **Convergence**: How quickly correction loop converges
- **Target Accuracy**: Final distance from target coordinates

### Training Command
```bash
# Train Vec2Text model
curl -X POST "http://localhost:8000/api/advanced-gap-filling/train-vec2text" \
  -H "Content-Type: application/json" \
  -d '{"epochs": 3, "batch_size": 8}'
```

## Combined Training Pipeline

### Recommended Training Order
1. **Initialize Service**: Load all documents with embeddings and coordinates
2. **Train Parametric UMAP**: Establish 2D↔embedding mapping
3. **Train Vec2Text**: Establish embedding→text generation
4. **Evaluate Combined Pipeline**: Test end-to-end gap filling
5. **Fine-tune**: Adjust parameters based on results

### Training Script
```bash
#!/bin/bash

echo "🚀 Starting Advanced Gap Filling Training Pipeline"

# Step 1: Initialize service
echo "📊 Initializing service with documents..."
curl -X POST "http://localhost:8000/api/advanced-gap-filling/initialize"

# Step 2: Train Parametric UMAP
echo "🗺️ Training Parametric UMAP..."
curl -X POST "http://localhost:8000/api/advanced-gap-filling/train-parametric-umap"

# Step 3: Train Vec2Text
echo "📝 Training Vec2Text..."
curl -X POST "http://localhost:8000/api/advanced-gap-filling/train-vec2text" \
  -H "Content-Type: application/json" \
  -d '{"epochs": 3}'

# Step 4: Evaluate quality
echo "📈 Evaluating model quality..."
curl -X GET "http://localhost:8000/api/advanced-gap-filling/evaluate-quality"

# Step 5: Save models
echo "💾 Saving trained models..."
curl -X POST "http://localhost:8000/api/advanced-gap-filling/save-parametric-umap"
curl -X POST "http://localhost:8000/api/advanced-gap-filling/save-vec2text"

echo "✅ Training pipeline completed!"
```

## Troubleshooting

### Common Issues

#### 1. Parametric UMAP Training Fails
- **Cause**: Insufficient data or memory issues
- **Solution**: Reduce batch size, use fewer samples, or increase memory

#### 2. Vec2Text Poor Quality
- **Cause**: Insufficient training data or poor embedding quality
- **Solution**: Increase training data, check embedding quality, adjust learning rate

#### 3. High Reconstruction Error
- **Cause**: Parametric UMAP not converged
- **Solution**: Increase epochs, adjust learning rate, or change architecture

#### 4. Slow Training
- **Cause**: Large dataset or complex model
- **Solution**: Reduce batch size, use fewer epochs, or simplify architecture

### Performance Optimization

#### For Large Datasets
- Use gradient checkpointing
- Reduce model complexity
- Use mixed precision training
- Implement data parallelism

#### For Better Quality
- Increase training epochs
- Use learning rate scheduling
- Implement early stopping
- Add regularization

## Monitoring Training

### Key Metrics to Watch
1. **Loss Curves**: Should decrease steadily
2. **Validation Metrics**: Should improve over time
3. **Memory Usage**: Should stay within limits
4. **Training Time**: Should be reasonable for dataset size

### Logging
All training progress is logged with detailed metrics. Check the console output for:
- Training progress
- Loss values
- Validation scores
- Error messages

## Next Steps After Training

1. **Test Gap Filling**: Try generating text at various map coordinates
2. **Evaluate Quality**: Check if generated text is semantically appropriate
3. **Fine-tune Parameters**: Adjust based on results
4. **Save Models**: Ensure models are properly saved for production use
5. **Update Documentation**: Record any parameter changes or improvements

## Production Considerations

- **Model Versioning**: Keep track of model versions and performance
- **Backup Models**: Always backup trained models
- **Monitoring**: Set up monitoring for model performance in production
- **Updates**: Plan for regular retraining with new data
