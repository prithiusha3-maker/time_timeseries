import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm

def evaluate(model, X_test, y_test, device=None, scaler=None, plot=True):
    if device is None:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import torch
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_test).float().to(device))
        if isinstance(pred, tuple):
            pred = pred[0]
        pred = pred.cpu().numpy()
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"MAE: {mae:.6f}, RMSE: {rmse:.6f}")
    if plot:
        plt.figure(figsize=(10,4))
        plt.plot(pred, label='Predicted')
        plt.plot(y_test, label='Actual', alpha=0.7)
        plt.legend()
        plt.title('Forecast vs Actual')
        plt.show()
    return pred, mae, rmse

def benchmark_sarimax(series, steps=100, order=(1,1,1), seasonal_order=(0,0,0,0)):
    # series: 1D numpy array or pandas Series
    try:
        mod = sm.tsa.SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        res = mod.fit(disp=False)
        pred = res.get_prediction(start=len(series), end=len(series)+steps-1)
        return pred.predicted_mean.values
    except Exception as e:
        print('SARIMAX benchmark failed:', e)
        return None
