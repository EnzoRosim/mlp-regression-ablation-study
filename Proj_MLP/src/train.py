import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import json
import copy
import matplotlib.pyplot as plt

from model import MLP
from dataset import carregar_dados_split, RAIZ_PROJETO

SEED = 42
torch.manual_seed(SEED)

def criar_dataloader(X, y, batch_size=32, embaralhar=True):
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=embaralhar)

def treinar_modelo(nome_experimento, X_treino, y_treino, X_val, y_val,
                    n_ocultas=(128, 64), dropout_p=0.0, l1_lambda=0.0, l2_weight_decay=0.0,
                    momentum=0.0, lr=0.01, batch_size=32,
                    n_epocas_max=1000, patience=50, epocas_minimas=150):

    n_entrada = X_treino.shape[1]
    modelo = MLP(n_entrada=n_entrada, n_ocultas=n_ocultas, dropout_p=dropout_p)

    criterio_mse = nn.MSELoss()
    criterio_mae = nn.L1Loss()
    otimizador = optim.SGD(modelo.parameters(), lr=lr,
                            momentum=momentum, weight_decay=l2_weight_decay)

    loader_treino = criar_dataloader(X_treino, y_treino, batch_size=batch_size)

    historico_treino, historico_val = [], []
    melhor_perda_val = float("inf")
    melhor_estado = None
    contador_paciencia = 0
    epoca_parada = n_epocas_max

    for epoca in range(n_epocas_max):
        modelo.train()
        for X_lote, y_lote in loader_treino:
            otimizador.zero_grad()
            saida = modelo(X_lote)
            perda = criterio_mse(saida, y_lote)

            if l1_lambda > 0:
                l1_penalidade = sum(p.abs().sum() for p in modelo.parameters())
                perda = perda + l1_lambda * l1_penalidade

            perda.backward()
            otimizador.step()

        # Avaliação em treino e validação inteiros (não em lotes), pra registrar a curva
        modelo.eval()
        with torch.no_grad():
            perda_treino = criterio_mse(modelo(X_treino), y_treino).item()
            perda_val = criterio_mse(modelo(X_val), y_val).item()

        historico_treino.append(perda_treino)
        historico_val.append(perda_val)

        # Early stopping
        if perda_val < melhor_perda_val:
            melhor_perda_val = perda_val
            melhor_estado = copy.deepcopy(modelo.state_dict())
            contador_paciencia = 0
        else:
            contador_paciencia += 1

        if epoca % 20 == 0:
            print(f"[{nome_experimento}] Época {epoca}: treino={perda_treino:.4f}, val={perda_val:.4f}, paciência={contador_paciencia}")

        if epoca >= epocas_minimas and contador_paciencia >= patience:
            epoca_parada = epoca
            print(f"[{nome_experimento}] Early stopping na época {epoca} (sem melhora por {patience} épocas)")
            break

    # Restaura o melhor modelo encontrado, não o último (que pode já estar overfitado)
    modelo.load_state_dict(melhor_estado)

    return modelo, historico_treino, historico_val, epoca_parada

def salvar_resultado(nome_experimento, historico_treino, historico_val, epoca_parada, config, metricas_teste):
    pasta_graficos = os.path.join(RAIZ_PROJETO, "outputs", "graficos")
    pasta_logs = os.path.join(RAIZ_PROJETO, "outputs", "logs")
    os.makedirs(pasta_graficos, exist_ok=True)
    os.makedirs(pasta_logs, exist_ok=True)

    plt.figure()
    plt.plot(historico_treino, label="Treino")
    plt.plot(historico_val, label="Validação")
    plt.xlabel("Época")
    plt.ylabel("Perda (MSE)")
    plt.title(nome_experimento)
    plt.legend()
    plt.savefig(os.path.join(pasta_graficos, f"{nome_experimento}.png"))
    plt.close()

    log = {
        "config": config,
        "epoca_parada": epoca_parada,
        "perda_treino_final": historico_treino[-1],
        "perda_val_final": historico_val[-1],
        "metricas_teste": metricas_teste,
    }
    with open(os.path.join(pasta_logs, f"{nome_experimento}.json"), "w") as f:
        json.dump(log, f, indent=2)

from metrics import calcular_metricas
from dataset import carregar_dados_split_normalizado

def avaliar_no_teste(modelo, X_teste, y_teste, scaler_y):
    modelo.eval()
    with torch.no_grad():
        y_pred_norm = modelo(X_teste)

    # Desfaz a normalização pra calcular métricas na escala real
    y_pred_real = scaler_y.inverse_transform(y_pred_norm.numpy())
    y_teste_real = scaler_y.inverse_transform(y_teste.numpy())

    y_pred_real = torch.tensor(y_pred_real)
    y_teste_real = torch.tensor(y_teste_real)

    return calcular_metricas(y_pred_real, y_teste_real)


if __name__ == "__main__":
    caminho_csv = os.path.join(RAIZ_PROJETO, "data", "dataset_projeto1.csv")
    pasta_splits = os.path.join(RAIZ_PROJETO, "outputs", "splits")

    X_treino, y_treino, X_val, y_val, X_teste, y_teste, scaler_y = carregar_dados_split_normalizado(
        caminho_csv, pasta_splits
    )

    # Lista de experimentos: nome + configuração
    # Baseline = todos os componentes zerados. Cada ablação muda SÓ um componente por vez.
    experimentos = [
        ("baseline", {"dropout_p": 0.0, "l1_lambda": 0.0, "l2_weight_decay": 0.0, "momentum": 0.0, "lr": 0.10}),
        ("momentum", {"dropout_p": 0.0, "l1_lambda": 0.0, "l2_weight_decay": 0.0, "momentum": 0.9, "lr": 0.01}),
        ("l2", {"dropout_p": 0.0, "l1_lambda": 0.0, "l2_weight_decay": 0.001, "momentum": 0.0, "lr": 0.10}),
        ("l1", {"dropout_p": 0.0, "l1_lambda": 0.0001, "l2_weight_decay": 0.0, "momentum": 0.0, "lr": 0.10}),
        ("dropout", {"dropout_p": 0.2, "l1_lambda": 0.0, "l2_weight_decay": 0.0, "momentum": 0.0, "lr": 0.10}),
        ("combinado", {"dropout_p": 0.0, "l1_lambda": 0.0, "l2_weight_decay": 0.001, "momentum": 0.9, "lr": 0.01}),
    ]

    resultados = {}  # guarda tudo pra facilitar comparação depois

    for nome, config in experimentos:
        print(f"\n{'='*50}")
        print(f"Treinando: {nome}")
        print(f"{'='*50}")

        modelo, hist_tr, hist_val, epoca_parada = treinar_modelo(
            nome, X_treino, y_treino, X_val, y_val,
            patience=50,
            epocas_minimas=150,
            **config
        )

        metricas_teste = avaliar_no_teste(modelo, X_teste, y_teste, scaler_y)
        print(f"Métricas no teste [{nome}]:", metricas_teste)

        salvar_resultado(nome, hist_tr, hist_val, epoca_parada, config, metricas_teste)

        resultados[nome] = {
            "modelo": modelo,
            "epoca_parada": epoca_parada,
            "metricas_teste": metricas_teste,
        }

    # Resumo final comparando todos os experimentos de uma vez
    print(f"\n{'='*50}")
    print("RESUMO COMPARATIVO")
    print(f"{'='*50}")
    for nome, dados in resultados.items():
        m = dados["metricas_teste"]
        print(f"{nome:12s} | época={dados['epoca_parada']:4d} | "
              f"MAE={m['MAE']:.4f} | MSE={m['MSE']:.4f} | "
              f"RMSE={m['RMSE']:.4f} | R²={m['R2']:.4f}")
    melhor_nome = max(resultados, key=lambda k: resultados[k]["metricas_teste"]["R2"])
    melhor_modelo = resultados[melhor_nome]["modelo"]
    print(f"\nMelhor modelo (maior R²): {melhor_nome}")

    melhor_modelo.eval()
    with torch.no_grad():
        y_pred_norm = melhor_modelo(X_teste)
    y_pred_real = scaler_y.inverse_transform(y_pred_norm.numpy())
    y_teste_real = scaler_y.inverse_transform(y_teste.numpy())

    pasta_graficos = os.path.join(RAIZ_PROJETO, "outputs", "graficos")

    # --- Parity Plot (Real vs. Previsto) ---
    plt.figure()
    plt.scatter(y_teste_real, y_pred_real, alpha=0.6, s=20)
    lims = [min(y_teste_real.min(), y_pred_real.min()), max(y_teste_real.max(), y_pred_real.max())]
    plt.plot(lims, lims, "r--", label="y = x (ideal)")
    plt.xlabel("Valor Real")
    plt.ylabel("Valor Previsto")
    plt.title(f"Parity Plot — {melhor_nome}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(pasta_graficos, "parity_plot.png"))
    plt.close()

    # --- Gráfico de Resíduos ---
    residuos = (y_teste_real - y_pred_real).flatten()
    plt.figure()
    plt.scatter(y_pred_real, residuos, alpha=0.6, s=20)
    plt.axhline(0, color="r", linestyle="--")
    plt.xlabel("Valor Previsto")
    plt.ylabel("Resíduo (Real - Previsto)")
    plt.title(f"Gráfico de Resíduos — {melhor_nome}")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(pasta_graficos, "residuos.png"))
    plt.close()

    print("Parity plot e gráfico de resíduos salvos em outputs/graficos/")