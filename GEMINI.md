# INF577 Quiz 2 - Facial Attribute Classification

## Project Overview
This project is a machine learning pipeline for facial attribute classification (e.g., age, gender, smiling). It uses PyTorch, Hydra for modular configuration, and Weights & Biases (W&B) for experiment tracking.

## Architecture & Development Standards

### 1. Modular Configuration (Hydra Config Groups)
The project uses Hydra's **Config Groups** to isolate component parameters. This allows switching architectures without modifying code.
- **Root Config**: `config/config.yaml` uses a `defaults` list to select the active components.
- **Component Configs**: Specific parameters for each backbone live in `config/backbone/*.yaml`.
- **Injection**: When a backbone is selected (e.g., `backbone: CNN`), its parameters are automatically available under `cfg.backbone`.

### 2. Model Factory Pattern
To keep `main.py` architecture-agnostic, all models must be instantiated via a **Factory**.
- **Location**: `models/backbone/factory.py`
- **Pattern**: The factory receives the component name and its specific configuration sub-object (`cfg.backbone`).
- **Benefit**: Adding a new architecture only requires registering it in the factory, keeping the training loop untouched.

### 3. Optimizer & Scheduler Modularization
Optimizers and Schedulers are managed via Hydra Config Groups for maximum flexibility.
- **Config**:
    - Optimizers: `config/optimizer/*.yaml` (e.g., Adam, SGD).
    - Schedulers: `config/scheduler/*.yaml` (e.g., CosineAnnealingLR, ReduceLROnPlateau).
- **Factory**: 
    - Optimizers: `models/optimizer.py`
    - Schedulers: `models/scheduler/factory.py`
- **Benefit**: You can swap training strategies (e.g., `scheduler=CosineAnnealingLR`) without changing the training loop in `main.py`.

### 4. Model Implementation
- Every backbone and head class must inherit from `nn.Module`.
- The constructor should accept a Hydra configuration object (`cfg`) to maintain the link between YAML definitions and implementation logic.
- **Architecture Flow**: All backbones should output a flattened feature vector, and heads should be designed to take that vector as input.

## How to Add a New Component (Backbone/Head/Optimizer)
1. **Config**: Create the corresponding `.yaml` in `config/<group>/`.
2. **Implementation**: Create the `.py` file (if applicable).
3. **Factory**: Register the new component in its respective factory.
4. **Execution**: Override via CLI: `python main.py optimizer=SGD optimizer.lr=0.01`.
## Training Pipeline

### 1. Data Handling (`data/dataset.py`)
- **Dataset Class**: `FacialAttributesDataset` loads `.npz` files, normalizes images to `[0, 1]`, and handles both training (with labels) and testing (images only) modes.
- **Data Splitting**: The training script automatically splits the input training file (e.g., `train_tiny.npz`) into 80% training and 20% validation sets to ensure reliable evaluation.

### 2. Training Loop (`main.py`)
- **Metric**: The primary evaluation metric is the **F1-Score**, calculated at the end of each validation epoch.
- **Loss Function**: `nn.BCEWithLogitsLoss` is used for binary classification.
- **Reproducibility**: `torch.manual_seed(42)` is set to ensure consistent results across runs.
- **Logging**: Training Loss, Validation Loss, and Validation F1-Score are logged to Weights & Biases.

### 3. Execution
To run a standard training session:
```bash
python main.py
```
To override training parameters:
```bash
python main.py training.epochs=20 training.batch_size=64
```

## How to Add a New Component (Backbone/Head/Optimizer)
...

- **Modularity**: Always use factories for component instantiation.
- **Separation of Concerns**: Keep architecture definitions in YAML and logic in Python.
- **Tracking**: All runs must be logged to `wandb`. Use `WANDB_MODE=offline` for local testing.
