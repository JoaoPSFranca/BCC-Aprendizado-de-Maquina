from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Carregando os dados
dados = load_iris()
X = dados.data

# Padronizando os dados
scaler = StandardScaler()
X_padronizado = scaler.fit_transform(X)

inercias = []
valores_k = range(1, 7)

for k in valores_k:
    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    modelo.fit(X_padronizado)
    inercias.append(modelo.inertia_)


plt.plot(valores_k, inercias, marker="o")
plt.axvline(x=3, color='red', linestyle='--', label='Melhor K (cotovelo)')
plt.plot(3, inercias[2], 'ro', markersize=10, label='K ótimo = 3')
plt.xlabel("Número de grupos — K")
plt.ylabel("Inércia")
plt.title("Método do cotovelo")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()