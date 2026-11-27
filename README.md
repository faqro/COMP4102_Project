# Robust Image Retrieval: Classical Features vs Deep Learning Embeddings

Faraaz Jan 101298165

COMP4102 Final Project comparing classical computer vision local feature descriptors (**SIFT**, **ORB**) and modern deep learning global image embeddings (**ResNet-50**, **DINOv2 ViT-B/14**) on the **Revisited Oxford5k (ROxford5k)** landmark retrieval benchmark.

## Setup

1. Create a Python virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

3. Run unit tests

```powershell
pytest tests/ -v
```

## Reproducing Experiments

```powershell
# Download ROxford5k and run baseline
python run_experiment.py --download --method all --experiment baseline

# Run validation-split parameter tuning and held-out test evaluation
python run_experiment.py --method all --experiment tuning

# Run query transformation robustness evaluation
python run_experiment.py --method all --experiment robustness

# Run gallery scalability and FAISS indexing comparison
python run_experiment.py --method all --experiment scalability
```

## Generating Figures

```powershell
python -c "from src.data.dataset import ROxford5kDataset; from src.visualization.plots import generate_all_figures; generate_all_figures(dataset=ROxford5kDataset())"
```