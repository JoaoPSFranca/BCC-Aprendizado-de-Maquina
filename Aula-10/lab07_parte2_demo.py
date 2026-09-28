"""
Aula 7 - Parte 2 (15/09/2026)
Modelos preditivos: classificacao - conceitos e fronteiras de decisao

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# ============================================================ BLOCO 1
# Dataset sintetico: duas classes em forma de "lua", que NAO podem ser
# separadas por uma unica reta - de proposito, para expor a diferenca
# entre fronteiras lineares e nao lineares.
X, y = make_moons(n_samples=300, noise=0.25, random_state=42)
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.30, random_state=42
)
print(f"Treino: {len(X_treino)} pontos   Teste: {len(X_teste)} pontos")
print(f"Classes: {sorted(set(y))} (0 e 1, sem significado de ordem)")

# ============================================================ BLOCO 2
# As 5 familias de classificadores apresentadas hoje.
modelos = {
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Regressao Logistica": LogisticRegression(),
    "Arvore de Decisao (max_depth=4)": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Naive Bayes (Gaussiano)": GaussianNB(),
    "SVM (kernel=rbf)": SVC(kernel="rbf", random_state=42),
}

for nome, modelo in modelos.items():
    modelo.fit(X_treino, y_treino)

# ============================================================ BLOCO 3
# Acuracia de cada modelo no MESMO conjunto de teste - so uma primeira
# leitura; validacao cruzada formal fica para a Semana 9 (28/09).
print("\nAcuracia no conjunto de teste (30% dos pontos, nunca vistos no treino):")
for nome, modelo in modelos.items():
    acc = accuracy_score(y_teste, modelo.predict(X_teste))
    print(f"  {nome:32s} acuracia = {acc:.3f}")

# ============================================================ BLOCO 4
# Um ponto "dificil": perto da fronteira entre as duas luas. Cada
# familia pode classifica-lo de um jeito diferente - e isso e esperado.
ponto_dificil = np.array([[0.5, 0.2]])
print(f"\nPonto de teste (dificil, perto da fronteira): {ponto_dificil[0]}")
for nome, modelo in modelos.items():
    pred = modelo.predict(ponto_dificil)[0]
    print(f"  {nome:32s} classifica como: {pred}")

# ============================================================ BLOCO 5
# Figura: a fronteira de decisao de cada familia, no MESMO dataset.
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),
    np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300),
)
grid = np.c_[xx.ravel(), yy.ravel()]

fig, eixos = plt.subplots(1, 5, figsize=(20, 4.3), sharey=True)
cores_pontos = ["#2c5aa0", "#c0392b"]
cmap_fundo = plt.cm.RdBu_r.reversed()  # azul de um lado, vermelho do outro

for ax, (nome, modelo) in zip(eixos, modelos.items()):
    Z = modelo.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=["#dbe7f6", "#f7d9d3"], alpha=0.9)
    ax.contour(xx, yy, Z, levels=[0.5], colors="#4c4c4c", linewidths=1.2)
    for classe, cor in zip([0, 1], cores_pontos):
        pts = X_teste[y_teste == classe]
        ax.scatter(pts[:, 0], pts[:, 1], c=cor, s=14, edgecolors="white", linewidths=0.4)
    ax.set_title(nome, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle(
    "Mesmos dados, 5 fronteiras de decisao diferentes (pontos = conjunto de teste)",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fronteiras_decisao_comparacao.png", dpi=150)
print("\nFigura salva em fronteiras_decisao_comparacao.png")
