# ML_elyte: Molecular Property Prediction Framework

A comprehensive machine learning framework for molecular property prediction using the QM9 dataset. ML_elyte provides tools for loading molecular data, extracting features, training both PyTorch and scikit-learn models, and evaluating performance with standard regression metrics.

## Features

- **Dataset Support**: QM9 molecular dataset with 12 quantum chemical properties
- **Feature Engineering**: Molecular descriptors, fingerprints, and physicochemical properties using RDKit
- **Dual Framework Support**: Both PyTorch deep learning models and scikit-learn traditional ML models
- **Comprehensive Metrics**: MAE, RMSE, R² evaluation for each property
- **Data Loading**: PyTorch DataLoader implementation for efficient batching
- **Multiple Architectures**: MLP, GNN, and Attention-based models
- **Visualization**: Built-in plotting for predictions, residuals, and model comparisons

## Installation

```bash
git clone https://github.com/M1chaY/ML_elyte.git
cd ML_elyte
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from ml_elyte import QM9Dataset, MolecularDataLoader, MLPRegressor, MLModel
from ml_elyte.metrics import RegressionMetrics
import torch

# Create dataset
dataset = QM9Dataset(
    data_dir="./data",
    target_properties=['mu', 'alpha', 'homo', 'lumo'],
    use_features=['descriptors', 'fingerprints']
)

# Create data loaders
data_loader = MolecularDataLoader(dataset, batch_size=32)
train_loader, val_loader, test_loader = data_loader.get_all_loaders()

# Train PyTorch model
model = MLPRegressor(
    input_dim=data_loader.get_feature_dim(),
    output_dim=data_loader.get_target_dim(),
    hidden_dims=[512, 256, 128]
)

# Train scikit-learn model
sklearn_model = MLModel(model_type='random_forest')
# ... training code ...

# Evaluate with comprehensive metrics
metrics = RegressionMetrics(property_names=['mu', 'alpha', 'homo', 'lumo'])
results = metrics.compute_metrics(y_true, y_pred)
metrics.print_metrics(results)
```

### Run the Demo

```bash
cd examples
python demo.py
```

This will create a complete example with dummy data, train multiple models, and generate comparison plots.

## Components

### Data Module (`ml_elyte.data`)

- **QM9Dataset**: PyTorch Dataset for QM9 molecular data
- **MolecularDataLoader**: DataLoader wrapper with proper batching and splitting

```python
from ml_elyte.data import QM9Dataset, MolecularDataLoader

dataset = QM9Dataset(
    data_dir="./qm9_data",
    target_properties=['mu', 'alpha', 'homo'],
    use_features=['descriptors', 'fingerprints']
)

loader = MolecularDataLoader(dataset, batch_size=64)
```

### Models Module (`ml_elyte.models`)

#### PyTorch Models
- **MLPRegressor**: Multi-layer perceptron with configurable architecture
- **GraphNeuralNetwork**: Basic GNN implementation (placeholder for graph-based models)
- **AttentionMLP**: MLP with multi-head attention

```python
from ml_elyte.models import MLPRegressor, get_model

# Direct instantiation
model = MLPRegressor(
    input_dim=1024, 
    output_dim=5,
    hidden_dims=[512, 256, 128],
    dropout_rate=0.2
)

# Factory method
model = get_model('mlp', input_dim=1024, output_dim=5, hidden_dims=[256, 128])
```

#### Scikit-learn Models
- **MLModel**: Unified wrapper for multiple sklearn regressors
- Support for Random Forest, Gradient Boosting, Ridge, Lasso, SVR, MLP

```python
from ml_elyte.models import MLModel

model = MLModel(
    model_type='random_forest',
    multi_output=True,
    scale_features=True
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Features Module (`ml_elyte.features`)

- **MolecularFeaturizer**: Extract molecular features from SMILES strings
- Support for descriptors, fingerprints, and physicochemical properties

```python
from ml_elyte.features import MolecularFeaturizer

featurizer = MolecularFeaturizer(
    feature_types=['descriptors', 'fingerprints'],
    fingerprint_radius=2,
    fingerprint_bits=1024
)

features = featurizer.featurize("CCO")  # Ethanol
```

### Metrics Module (`ml_elyte.metrics`)

- **RegressionMetrics**: Comprehensive evaluation metrics and visualization

```python
from ml_elyte.metrics import RegressionMetrics

metrics = RegressionMetrics(property_names=['mu', 'alpha'])
results = metrics.compute_metrics(y_true, y_pred, per_property=True)
metrics.print_metrics(results)
metrics.plot_predictions(y_true, y_pred, property_idx=0)
```

## QM9 Properties

The framework supports all 12 QM9 properties:

| Property | Unit | Description |
|----------|------|-------------|
| mu | Debye | Dipole moment |
| alpha | Bohr³ | Isotropic polarizability |
| homo | Hartree | Highest occupied molecular orbital energy |
| lumo | Hartree | Lowest unoccupied molecular orbital energy |
| gap | Hartree | HOMO-LUMO gap |
| r2 | Bohr² | Electronic spatial extent |
| zpve | Hartree | Zero point vibrational energy |
| u0 | Hartree | Internal energy at 0K |
| u298 | Hartree | Internal energy at 298.15K |
| h298 | Hartree | Enthalpy at 298.15K |
| g298 | Hartree | Free energy at 298.15K |
| cv | cal/mol/K | Heat capacity at 298.15K |

## Examples

### Training Scripts

1. **PyTorch Training**:
   ```bash
   python examples/train_pytorch_model.py --model_type mlp --epochs 50 --batch_size 64
   ```

2. **Scikit-learn Training**:
   ```bash
   python examples/train_sklearn_models.py --models random_forest gradient_boosting --hyperparameter_search
   ```

### Data Formats

The framework supports multiple molecular data formats:
- **SMILES**: Text representation of molecular structures
- **SDF**: Structure Data Format with 3D coordinates
- **CSV**: Tabular data with SMILES and properties
- **Pickle**: Serialized Python objects

## Architecture

```
ml_elyte/
├── data/              # Dataset and data loading utilities
├── models/            # PyTorch and scikit-learn models
├── features/          # Molecular feature extraction
├── metrics/           # Evaluation metrics and visualization
├── utils/             # Utility functions
└── examples/          # Example scripts and demos
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- scikit-learn 1.3+
- RDKit (via rdkit-pypi)
- NumPy, pandas, matplotlib, seaborn

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - see LICENSE file for details.

## Citation

If you use ML_elyte in your research, please cite:

```bibtex
@software{ml_elyte,
  title={ML_elyte: Molecular Property Prediction Framework},
  author={ML_elyte Contributors},
  year={2024},
  url={https://github.com/M1chaY/ML_elyte}
}
```