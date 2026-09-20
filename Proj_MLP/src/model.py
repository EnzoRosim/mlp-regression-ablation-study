import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_entrada, n_ocultas=(64,32), dropout_p=0.0):
        super().__init__()
        camadas = []
        anterior = n_entrada
        for n in n_ocultas:
            camadas.append(nn.Linear(anterior, n))
            camadas.append(nn.ReLU())
            if dropout_p > 0:
                camadas.append(nn.Dropout(dropout_p))
            anterior = n
        camadas.append(nn.Linear(anterior, 1))
        self.rede = nn.Sequential(*camadas)

    def forward(self, x):
        return self.rede(x)