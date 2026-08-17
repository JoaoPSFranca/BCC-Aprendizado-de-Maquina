
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

SEMENTE = 42
pd.set_option("display.width", 100)


def titulo(n, texto):
    print()
    print("=" * 70)
    print(f"BLOCO {n} - {texto}")
    print("=" * 70)


# ==================================================================== BLOCO 1
titulo(1, "RELEMBRANDO: O CAMINHO SUPERVISIONADO (K-NN)")

dados = load_iris()
X = dados.data
y = dados.target

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.30, random_state=SEMENTE, stratify=y,
)
knn = KNeighborsClassifier(n_neighbors=5).fit(X_treino, y_treino)
ac = accuracy_score(y_teste, knn.predict(X_teste))
print(f"K-NN usou y_treino para aprender e y_teste para avaliar.")
print(f"Acuracia no teste: {ac:.4f}")

# ==================================================================== BLOCO 2
titulo(2, "O CAMINHO NAO SUPERVISIONADO (K-MEANS)")

print("Agora fingimos que y nao existe. So X entra no algoritmo.")
#n_clusters = quantidade de gruposa formar
#random_state = fixa a aleatoriedade
#n_init = executa po K_means 10x com centroides iniciais escolhendo a melhor
kmeans = KMeans(n_clusters=3, random_state=SEMENTE, n_init=10)

#Essa linha realiza duas operações:
#fit: encontra os três centroides usando os dados de X.
#predict: atribui cada flor ao centroide mais próximo.
clusters = kmeans.fit_predict(X)
print(clusters)

print("\nPrimeiros 10 grupos atribuidos:", clusters[:10])
print("(Sao rotulos de GRUPO -0,1,2 - nao correspondem necessariamente a")
print(" 0=setosa, 1=versicolor, 2=virginica: o algoritmo nao sabe os nomes.)")

# ==================================================================== BLOCO 3
titulo(3, "COMPARANDO COM y (SO POSSIVEL PORQUE E UM DATASET DIDATICO)")

tabela = pd.crosstab(pd.Series(y, name="real"), pd.Series(clusters, name="cluster"))
print(tabela)
print("A leitura seria:")
print("As 50 flores da classe real 0 foram colocadas no cluster 1.")
print("A classe real 1 ficou principalmente no cluster 0.")
print("A classe real 2 ficou dividida entre os clusters 0 e 2.")

ari = adjusted_rand_score(y, clusters)
print(f"\nAdjusted Rand Index (ARI): {ari:.4f}")
print("""
LEITURA:
  ARI = 1  -> os grupos encontrados coincidem perfeitamente com as classes.
  ARI = 0  -> os grupos nao tem relacao melhor que o acaso com as classes.
  ARI negativo -> pior que o acaso (raro na pratica).

  Aqui, setosa cai inteira em um unico grupo, sem mistura - a mesma
  separacao nitida que o K-NN ja explorava desde a Aula 1. Entre
  versicolor e virginica o agrupamento se confunde, pelo mesmo motivo de
  sempre: as duas especies se sobrepoem no espaco de atributos.""")

# ==================================================================== BLOCO 4
titulo(4, "RESUMO: DOIS CAMINHOS, MESMO X")

print(f"""
  Caminho           | usa y no treino? | usa y na avaliacao? | metrica
  -------------------+-------------------+----------------------+------------------
  K-NN (Bloco 1)      | SIM               | SIM                  | acuracia = {ac:.4f}
  K-Means (Bloco 2-3) | NAO               | so por curiosidade   | ARI = {ari:.4f}

Em um problema nao supervisionado real, a linha "so por curiosidade" nao
existe: voce nunca teria y para comparar. A pergunta "quantos grupos
existem, e sao bons?" .

FIM DA DEMONSTRACAO.
Abra agora o lab03_parte1_exercicios.py e resolva os TODO (30 min).""")
