# src/busca_baseline.py
import os
from train import treinar_modelo
from dataset import carregar_dados_split_normalizado, RAIZ_PROJETO

caminho_csv = os.path.join(RAIZ_PROJETO, "data", "dataset_projeto1.csv")
pasta_splits = os.path.join(RAIZ_PROJETO, "outputs", "splits")

X_treino, y_treino, X_val, y_val, X_teste, y_teste, scaler_y = carregar_dados_split_normalizado(
    caminho_csv, pasta_splits
)

arquiteturas = [(32,), (64, 32), (128, 64)]
learning_rates = [0.01, 0.03, 0.05, 0.1]

resultados_busca = []

for arq in arquiteturas:
    for lr in learning_rates:
        nome = f"busca_arq{arq}_lr{lr}"
        _, hist_tr, hist_val, epoca_parada = treinar_modelo(
            nome, X_treino, y_treino, X_val, y_val,
            n_ocultas=arq, lr=lr,
            momentum=0.0, dropout_p=0.0, l1_lambda=0.0, l2_weight_decay=0.0,  # baseline puro
            patience=50, epocas_minimas=150
        )
        melhor_val = min(hist_val)
        resultados_busca.append({
            "arquitetura": arq, "lr": lr,
            "melhor_perda_val": melhor_val, "epoca_parada": epoca_parada
        })

# Ordena do melhor (menor perda de validação) pro pior
resultados_busca.sort(key=lambda r: r["melhor_perda_val"])

print("\n=== RESULTADO DA BUSCA (ordenado por melhor perda de validação) ===")
for r in resultados_busca:
    print(f"arq={str(r['arquitetura']):15s} lr={r['lr']:.2f} | "
          f"perda_val={r['melhor_perda_val']:.4f} | época={r['epoca_parada']}")