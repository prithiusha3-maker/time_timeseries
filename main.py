import torch
from data_prep import generate_synthetic_dataset, scale_dataset, create_sequences
from models import LSTMAttentionModel, TransformerModel
from train import train_model
from evaluate import evaluate, benchmark_sarimax
import numpy as np

if __name__ == "__main__":
    # 1. Generate dataset
    df = generate_synthetic_dataset()
    data, scaler = scale_dataset(df)
    
    # 2. Create sequences
    X, y = create_sequences(data, window=50)

    # 3. Train-test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 4. Model initialization
    model = LSTMAttentionModel(input_dim=X.shape[2], hidden_dim=64)

    # 5. Train model
    trained = train_model(
        model, 
        X_train, y_train, 
        X_val=X_test, y_val=y_test,
        epochs=10, 
        lr=0.001
    )

    # 6. Evaluate deep learning model
    pred, mae, rmse = evaluate(trained, X_test, y_test)
    print("Final MAE:", mae)
    print("Final RMSE:", rmse)

    # 7. SARIMAX Benchmarking
    sarimax_pred = benchmark_sarimax(df["value1"].values, steps=len(y_test))
    print("SARIMAX benchmark completed.")
