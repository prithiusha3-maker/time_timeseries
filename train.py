import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        out = model(Xb)
        # model may return tuple (pred, weights)
        if isinstance(out, tuple):
            out = out[0]
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

def train_model(model, X_train, y_train, X_val=None, y_val=None, epochs=10, lr=1e-3, batch_size=32, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        loss = train_epoch(model, loader, criterion, optimizer, device)
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_pred = model(torch.tensor(X_val).float().to(device))
                if isinstance(val_pred, tuple):
                    val_pred = val_pred[0]
                val_pred = val_pred.cpu().numpy()
            val_mae, val_rmse = evaluate_predictions(y_val, val_pred)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {loss:.4f} - Val MAE: {val_mae:.4f} - Val RMSE: {val_rmse:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {loss:.4f}")
    return model

def objective(trial, model_class, X_train, y_train, X_val, y_val, device):
    # simple Optuna search space
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    hidden = trial.suggest_categorical('hidden', [32, 64, 128])
    batch = trial.suggest_categorical('batch', [16, 32, 64])
    model = model_class(input_dim=X_train.shape[2], hidden_dim=hidden)
    model = train_model(model, X_train, y_train, X_val=X_val, y_val=y_val, epochs=5, lr=lr, batch_size=batch, device=device)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_val).float().to(device))
        if isinstance(pred, tuple):
            pred = pred[0]
        pred = pred.cpu().numpy()
    mae, rmse = evaluate_predictions(y_val, pred)
    return rmse

def run_hyperparameter_search(model_class, X_train, y_train, X_val, y_val, n_trials=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, model_class, X_train, y_train, X_val, y_val, device), n_trials=n_trials)
    print('Best params:', study.best_params)
    return study.best_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning (Optuna)')
    args = parser.parse_args()
    # This file provides train_model and tuning utilities; actual invocation happens from main.py
