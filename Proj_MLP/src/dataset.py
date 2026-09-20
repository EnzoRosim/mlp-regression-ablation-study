from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os

def gerar_e_salvar_split(caminho_csv, pasta_saida, seed=42):
    df = pd.read_csv(caminho_csv)
    indices = np.arange(len(df))

    idx_temp, idx_teste = train_test_split(indices, test_size=0.8, random_state=seed)
    idx_treino, idx_val = train_test_split(idx_temp, test_size=0.5, random_state=seed)

    os.makedirs(pasta_saida, exist_ok=True)

    np.save(os.path.join(pasta_saida, "idx_treino.npy"), idx_treino)
    np.save(os.path.join(pasta_saida, "idx_val.npy"), idx_val)
    np.save(os.path.join(pasta_saida, "idx_teste.npy"), idx_teste)
    print(f"Split salvo: treino={len(idx_treino)}, val={len(idx_val)}, teste={len(idx_teste)}")


RAIZ_PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    caminho_csv = os.path.join(RAIZ_PROJETO, "data", "dataset_projeto1.csv")
    pasta_saida = os.path.join(RAIZ_PROJETO, "outputs", "splits")
    gerar_e_salvar_split(caminho_csv, pasta_saida)

import torch
from sklearn.preprocessing import StandardScaler

def carregar_dados_split(caminho_csv, pasta_splits):
    df = pd.read_csv(caminho_csv)

    idx_treino = np.load(os.path.join(pasta_splits, "idx_treino.npy"))
    idx_val = np.load(os.path.join(pasta_splits, "idx_val.npy"))
    idx_teste = np.load(os.path.join(pasta_splits, "idx_teste.npy"))

    X = df["x"].values.astype("float32").reshape(-1, 1)
    y = df["y"].values.astype("float32").reshape(-1, 1)

    X_treino = torch.tensor(X[idx_treino])
    y_treino = torch.tensor(y[idx_treino])
    X_val = torch.tensor(X[idx_val])
    y_val = torch.tensor(y[idx_val])
    X_teste = torch.tensor(X[idx_teste])
    y_teste = torch.tensor(y[idx_teste])

    return X_treino, y_treino, X_val, y_val, X_teste, y_teste

def carregar_dados_split_normalizado(caminho_csv, pasta_splits):
    df = pd.read_csv(caminho_csv)

    idx_treino = np.load(os.path.join(pasta_splits, "idx_treino.npy"))
    idx_val = np.load(os.path.join(pasta_splits, "idx_val.npy"))
    idx_teste = np.load(os.path.join(pasta_splits, "idx_teste.npy"))

    X = df["x"].values.astype("float32").reshape(-1, 1)
    y = df["y"].values.astype("float32").reshape(-1, 1)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    scaler_X.fit(X[idx_treino])
    scaler_y.fit(y[idx_treino])

    X_norm = scaler_X.transform(X)
    y_norm = scaler_y.transform(y)

    X_treino = torch.tensor(X_norm[idx_treino], dtype=torch.float32)
    y_treino = torch.tensor(y_norm[idx_treino], dtype=torch.float32)
    X_val = torch.tensor(X_norm[idx_val], dtype=torch.float32)
    y_val = torch.tensor(y_norm[idx_val], dtype=torch.float32)
    X_teste = torch.tensor(X_norm[idx_teste], dtype=torch.float32)
    y_teste = torch.tensor(y_norm[idx_teste], dtype=torch.float32)

    return X_treino, y_treino, X_val, y_val, X_teste, y_teste, scaler_y