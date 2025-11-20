# Advanced Time Series Forecasting with Deep Learning and Attention Mechanisms

## ✔ Models Implemented
- LSTM
- LSTM + Attention Mechanism
- Transformer Encoder Model

## ✔ Features
- Multivariate synthetic dataset
- Train/validation/test split
- Comparative forecasting results (including SARIMAX baseline)
- Attention weight extraction and visualization
- Hyperparameter tuning using Optuna (optional)
- Fully modular Python code

## ✔ How to Run (Local / Colab)
1. Install dependencies (if running locally):
```bash
pip install -r requirements.txt
```

2. Run the full pipeline:
```bash
python main.py
```

3. Optional: run hyperparameter tuning (uses Optuna):
```bash
python train.py --tune
```

## Files
- `data_prep.py` - dataset generation & preprocessing
- `models.py` - model definitions (LSTM, LSTM+Attention, Transformer)
- `train.py` - training loop, optional Optuna tuning
- `evaluate.py` - evaluation metrics, plotting, SARIMAX benchmarking
- `main.py` - orchestrates the full pipeline

## Author
Your Name
