# Deep Learning for Handwritten Mathematical Expression Recognition

This project implements three neural network architectures for handwritten mathematical expression recognition tasks, completed as part of the **COMP47650 Deep Learning** module at University College Dublin.

## Overview

The project tackles progressive complexity in sequence recognition:
1. **Part 1**: Glyph Classification (CNN)
2. **Part 2**: Infix Expression Recognition (RNN Encoder-Decoder)
3. **Part 3**: Postfix Expression Recognition (Transformer Decoder)

## Student Contributions

The following model architectures and training implementations were completed independently:

### Part 1: CNN Glyph Classifier
- **Architecture**: 5-layer 1D CNN with batch normalization and dropout
- **Key Features**:
  - He weight initialization for stable training
  - Progressive channel expansion (8→16→24→32→40)
  - Strategic dropout placement (0.1) to prevent overfitting
  - Confusion matrix analysis for per-class performance
- **Performance**: Achieved strong classification accuracy across 18 glyph classes
- **Comparison Model**: Alternative architecture using Layer Normalization and GELU activation with stride-based downsampling

### Part 2: RNN Encoder-Decoder
- **Architecture**: Custom GRU-based sequence-to-sequence model
- **Key Features**:
  - CNN pre-encoder for stroke feature extraction
  - Manual GRU gate implementation (reset, update, candidate hidden states)
  - Teacher forcing strategy with configurable ratio
  - Gradient clipping (max_norm=1.0) for training stability
- **Performance**: Effective infix expression recognition with Levenshtein Accuracy and Character Error Rate (CER) metrics
- **Comparison Model**: PyTorch nn.GRU-based implementation for architectural comparison

### Part 3: Transformer Decoder
- **Architecture**: 3-block transformer with multi-head self-attention (4 heads)
- **Key Features**:
  - Custom multi-head attention implementation
  - Pre-normalization transformer blocks (Pre-LN)
  - Causal masking for autoregressive generation
  - Combined stroke and token positional encoding
  - Greedy decoding for inference
- **Current Status**: Implementation complete with architectural challenges
- **Known Issues**: 
  - Training instability during extended epochs
  - Suboptimal convergence on complex postfix sequences
  - Attention mechanism may require additional regularization or architectural refinement
- **Comparison Model**: PyTorch nn.TransformerDecoder-based implementation

## Architecture Highlights

### Design Decisions
- **Weight Initialization**: He initialization used consistently across all models to address vanishing/exploding gradients
- **Normalization**: Batch normalization (CNN, RNN) vs Layer normalization (Transformer)
- **Dropout Strategy**: Incremental application in deeper layers to balance regularization and capacity
- **Sequence Processing**: Progressive complexity from fixed-length (CNN) to variable-length with attention (Transformer)

### Training Strategies
- Checkpoint-based training with resume capability
- Best validation accuracy tracking
- Per-epoch progress monitoring with tqdm
- Token-level accuracy calculation excluding padding

## Project Structure

```
.
├── models/
│   ├── part1_glyph_model.py          # CNN implementation
│   ├── part2_infix_model.py          # RNN encoder-decoder
│   └── part3_postfix_model.py        # Transformer decoder
├── scripts/
│   ├── part1_preprocessing.py
│   ├── part2_preprocessing.py
│   ├── part3_preprocessing.py
│   └── utils.py
├── notebooks/
│   ├── part1_startup.ipynb
│   ├── part2_startup.ipynb
│   ├── part3_startup.ipynb
│   └── evaluation.ipynb
├── checkpoints/                       # Model checkpoints
├── datasets/                          # Datasets (not included)
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- PyTorch >= 2.1.0
- torchvision >= 0.16.0
- h5py >= 3.8.0
- tqdm >= 4.65.0
- matplotlib >= 3.7.0
- torchinfo >= 1.8.0
- numpy >= 1.24.0
- editdistance == 0.8.1

## Usage

Each part includes a startup notebook for training and evaluation:

```python
# Example: Training Part 1 CNN
from models.part1_glyph_model import part1_glyph_classification_model, part1_train_model

model = part1_glyph_classification_model(**model_args)
history = part1_train_model(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    num_epochs=50,
    lr=1e-3,
    device="cuda",
    save_path="checkpoints/part1_best.pt"
)
```

## Note on Datasets

**Datasets are not included in this repository** due to size constraints and academic integrity policies. The project was developed using proprietary datasets provided by the COMP47650 module.

## Acknowledgments

This project was completed as coursework for **COMP47650 Deep Learning** at University College Dublin. The course framework, dataset preprocessing utilities, evaluation metrics, and project structure were provided by the module instructors. The neural network architectures, forward/backward propagation implementations, and training strategies represent student work.

**Course Framework Contributions:**
- Dataset preprocessing utilities (`scripts/utils.py`, preprocessing files)
- Evaluation metrics (Levenshtein Accuracy, batch_LA function)
- Test functions (`part*_test_model`)
- Project structure and notebook templates

**Student Implementation:**
- All model architectures (CNN, RNN, Transformer)
- Training loop implementations
- Custom GRU and attention mechanisms
- Weight initialization strategies
- Comparison model variants

## License

This project is submitted as academic coursework. Please respect academic integrity policies.

---

**Module**: COMP47650 Deep Learning  
**Institution**: University College Dublin  
**Year**: 2026
