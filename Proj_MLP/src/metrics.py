import torch

def calcular_metricas(y_pred, y_real):
    erro = y_pred - y_real

    mae = erro.abs().mean().item()
    mse = (erro ** 2).mean().item()
    rmse = mse ** 0.5

    ss_res = (erro ** 2).sum()
    ss_tot = ((y_real - y_real.mean()) ** 2).sum()
    r2 = (1 - ss_res / ss_tot).item()

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}