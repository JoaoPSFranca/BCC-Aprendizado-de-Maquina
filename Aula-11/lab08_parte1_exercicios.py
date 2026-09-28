"""
Aprendizado de Maquina - IFSP Presidente Epitacio
Aula 8, Parte 1 - MINI-LABORATORIO  ***ARQUIVO DO ALUNO***

"""
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
SEMENTE = 42
pd.set_option("display.width", 100)

def secao(n, titulo):
    print()
    print("=" * 70)
    print(f"EXERCICIO {n} - {titulo}")
    print("=" * 70)

def gerar_dados_emprestimo(n=450, semente=42):
    """Mesmo dataset das aulas de preparacao de dados: solicitacoes de
    emprestimo, com valores ausentes, uma coluna categorica e outliers."""
    rng = np.random.default_rng(semente)

    idade = rng.normal(38, 12, n).clip(18, 75).round().astype(float)
    score_credito = rng.normal(650, 80, n).clip(300, 850).round().astype(float)
    tempo_emprego = rng.normal(7, 5, n).clip(0, 35).round().astype(float)

    renda = rng.lognormal(mean=8.15, sigma=0.45, size=n)
    idx_outlier = rng.choice(n, size=5, replace=False)
    renda[idx_outlier] *= rng.uniform(6, 10, size=5)
    renda = renda.round(2)

    cidades = np.array(["SP", "RJ", "MG", "Outra"])
    cidade = rng.choice(cidades, size=n, p=[0.42, 0.23, 0.15, 0.20])

    renda_c = np.clip(renda, None, np.percentile(renda, 95))
    z = (
        0.05 * (score_credito - 650)
        + 0.0008 * (renda_c - renda_c.mean())
        + 0.12 * (tempo_emprego - 7)
        + rng.normal(0, 0.35, n)
    )
    prob_aprovado = 1 / (1 + np.exp(-z))
    aprovado = (rng.uniform(0, 1, n) < prob_aprovado).astype(int)

    df = pd.DataFrame({
        "idade": idade,
        "renda": renda,
        "tempo_emprego": tempo_emprego,
        "score_credito": score_credito,
        "cidade": cidade,
        "aprovado": aprovado,
    })

    mask_idade = rng.uniform(0, 1, n) < 0.08
    mask_renda = rng.uniform(0, 1, n) < 0.10
    mask_cidade = rng.uniform(0, 1, n) < 0.06
    df.loc[mask_idade, "idade"] = np.nan
    df.loc[mask_renda, "renda"] = np.nan
    df.loc[mask_cidade, "cidade"] = None
    return df

COL_NUM = ["idade", "renda", "tempo_emprego", "score_credito"]
COL_CAT = ["cidade"]

def montar_preprocessador(escalar=True):
    """Imputacao + (escala) nos numericos; imputacao + one-hot na categorica.
    Tudo dentro de um ColumnTransformer: as estatisticas (mediana, media,
    desvio, categorias) sao aprendidas SO no treino."""
    passos_num = [("imp", SimpleImputer(strategy="median"))]
    if escalar:
        passos_num.append(("esc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(passos_num), COL_NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), COL_CAT),
    ])

def criar_modelos():
    """Os 5 classificadores da demonstracao + random forest (6o modelo)."""
    return {
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Regressao Logistica": LogisticRegression(max_iter=1000),
        "Arvore de Decisao (max_depth=4)": DecisionTreeClassifier(max_depth=4, random_state=SEMENTE),
        "Naive Bayes (Gaussiano)": GaussianNB(),
        "SVM (kernel=rbf)": SVC(kernel="rbf", random_state=SEMENTE),
        "Random Forest (100 arvores)": RandomForestClassifier(n_estimators=100, random_state=SEMENTE),
    }

def metricas(y_verd, y_prev):
    return {
        "acurácia": accuracy_score(y_verd, y_prev),
        "precisão": precision_score(y_verd, y_prev, zero_division=0),
        "recall": recall_score(y_verd, y_prev, zero_division=0),
        "f1": f1_score(y_verd, y_prev, zero_division=0),
    }

# ============================================================ EXERCICIO 1 (1,5)
secao(1, "OS DADOS E O PISO (BASELINE)")

# TODO 1a: gere o dataset com gerar_dados_emprestimo(n=450, semente=SEMENTE)
#          e separe X (todas as colunas menos "aprovado") e y (a coluna "aprovado")

df = gerar_dados_emprestimo(n=450, semente=SEMENTE)
X = df.drop(columns="aprovado")
y = df["aprovado"]

# TODO 1b: divida em treino e teste com train_test_split(test_size=0.25,
#          random_state=SEMENTE, stratify=y) e imprima o tamanho de cada parte

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.30, random_state=SEMENTE, stratify=y
)

# TODO 1c: treine um DummyClassifier(strategy="most_frequent") no treino e
#          imprima sua acuracia no teste (esse e o "piso" da comparacao)
baseline = DummyClassifier(strategy="most_frequent").fit(X_treino, y_treino)
acc_base = accuracy_score(y_teste, baseline.predict(X_teste))

# ============================================================ EXERCICIO 2 (2,5)
secao(2, "OS 6 MODELOS NO MESMO TREINO E TESTE")

# TODO 2a: para cada modelo de criar_modelos(), monte um Pipeline com dois
#          passos: ("prep", montar_preprocessador()) e ("modelo", modelo).
#          Treine no treino e guarde as previsoes no teste.
resultados, predicoes = {}, {}
maxi = 0
melhor_modelo = None
nome_modelo = None
for nome, modelo in criar_modelos().items():
    pipe = Pipeline([("prep", montar_preprocessador()), ("modelo", modelo)])
    pipe.fit(X_treino, y_treino)
    predicoes[nome] = pipe.predict(X_teste)
    m = metricas(y_teste, predicoes[nome])
    resultados[nome] = m

    if maxi < m["f1"]:
        maxi = m["f1"]
        melhor_modelo = modelo
        nome_modelo = nome

# TODO 2b: monte uma tabela (DataFrame) com uma linha por modelo e as
#          colunas acuracia, precisao, recall e f1 (classe positiva = 1,
#          aprovado). Imprima a tabela arredondada em 3 casas.

tabela = pd.DataFrame(resultados).T.round(3)
tabela.loc["Baseline (classe mais comum)"] = [acc_base, np.nan, 0.0, 0.0]

# TODO 2c: imprima o nome do modelo com o MAIOR F1
print(f"MELHOR MODELO: {nome_modelo} com F1 = {maxi:.3f}")

# ============================================================ EXERCICIO 3 (2,0)
secao(3, "MATRIZ DE CONFUSAO DO MELHOR MODELO")

# TODO 3a: calcule a matriz de confusao do modelo com maior F1 (Exercicio 2c)
#          e extraia VN, FP, FN, VP com  tn, fp, fn, tp = cm.ravel()
fig, eixos = plt.subplots(1, 5, figsize=(16, 3.3))
for ax, (nome, y_prev) in zip(eixos, predicoes.items()):
    cm = confusion_matrix(y_teste, y_prev)
    ax.imshow(cm, cmap="Greens")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["negado", "aprovado"]); ax.set_yticklabels(["negado", "aprovado"])
    ax.set_xlabel("previsto"); ax.set_ylabel("real")
    ax.set_title(nome.split(" (")[0], fontsize=10)
    tn, fp, fn, tp = cm.ravel()
    precisao = tp / (tp + fn)
    recall = tp / (tp + fn)
    f1 = 2 * precisao * recall / (precisao + recall)
    print(f"{nome:34s} aprovações indevidas (FP) = {fp:2d} | negações indevidas (FN) = {fn:2d}")
    print(f"Precisao = {precisao:.3f} | Recall = {recall:.3f} | F1 = {f1:.3f}")

plt.tight_layout()
plt.show()

# TODO 3b: calcule NA MAO (so com tp, fp, fn) a precisao e o recall e
#          compare com os valores da tabela do Exercicio 2



# ============================================================ EXERCICIO 4 (2,0)
secao(4, "SEM PADRONIZAR: QUEM SOFRE?")

# TODO 4: repita o treino do k-NN, do SVM e da Arvore de Decisao usando
#         montar_preprocessador(escalar=False) (sem StandardScaler).
#         Imprima, para cada um, a acuracia COM escala, SEM escala e a
#         diferenca entre elas.

print("k-NN — variando o número de vizinhos (k):")
linhas = []
for k in [1, 3, 5, 15, 31, 71]:
    pipe = Pipeline([("prep", montar_preprocessador()), ("modelo", KNeighborsClassifier(k))]).fit(X_treino, y_treino)
    linhas.append((k, pipe.score(X_treino, y_treino), pipe.score(X_teste, y_teste)))
print(pd.DataFrame(linhas, columns=["k", "acc_treino", "acc_teste"]).round(3))

print("Árvore de decisão — variando a profundidade máxima:")
linhas = []
for d in [1, 2, 4, 8, 16, None]:
    pipe = Pipeline([("prep", montar_preprocessador()),
                     ("modelo", DecisionTreeClassifier(max_depth=d, random_state=SEMENTE))]).fit(X_treino, y_treino)
    linhas.append((str(d), pipe.score(X_treino, y_treino), pipe.score(X_teste, y_teste)))
print(pd.DataFrame(linhas, columns=["max_depth", "acc_treino", "acc_teste"]).round(3))

# ============================================================ EXERCICIO 5 (2,0)
secao(5, "CINCO SORTEIOS, CINCO CAMPEOES?")

# TODO 5a: para semente em range(5), refaca a divisao treino/teste
#           (test_size=0.25, random_state=semente, stratify=y), treine os 6
#           modelos e guarde a acuracia de cada um no teste
nomes = list(criar_modelos().keys())
acuracias = {n: [] for n in nomes}
vencedores = []
for semente in range(5):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=semente, stratify=y)
    acc_rodada = {}

    for nome, modelo in criar_modelos().items():
        pipe = Pipeline([("prep", montar_preprocessador()), ("modelo", modelo)]).fit(X_tr, y_tr)
        acc_rodada[nome] = pipe.score(X_te, y_te)
        acuracias[nome].append(acc_rodada[nome])

    vencedores.append(max(acc_rodada, key=acc_rodada.get))

# TODO 5b: para cada sorteio, descubra quem foi o campeao (maior acuracia).
#           Em caso de EMPATE, todos os empatados contam como campeoes.
#           Dica: tabela.eq(tabela.max(axis=1), axis=0)
resumo = pd.DataFrame({
    "media": {n: np.mean(v) for n, v in acuracias.items()},
    "desvio": {n: np.std(v) for n, v in acuracias.items()},
    "minimo": {n: np.min(v) for n, v in acuracias.items()},
    "maximo": {n: np.max(v) for n, v in acuracias.items()},
}).round(3)

print(resumo.to_string())
print("\nQuem foi o 1º colocado em cada um dos 5 sorteios:")
print(pd.Series(vencedores).value_counts().to_string())

# TODO 5c: imprima quantas vezes (de 5) cada modelo foi campeao e a
#          amplitude (maximo - minimo) da acuracia de cada modelo


# ===========================RELATORIO=================================
"""
=======================================================================
RELATORIO 
=======================================================================

Ex.1 - Qual a acuracia do baseline e o que ela significa? Por que o
       parametro stratify=y foi usado na divisao?
    R:

Ex.2 - Qual modelo teve o maior F1? A diferenca dele para o segundo
       colocado e grande ou pequena? Com um teste de 113 exemplos, voce
       diria que essa diferenca e confiavel? Justifique.
    R:

Ex.3 - No problema de emprestimo, qual erro e mais caro para o banco:
       aprovacao indevida (FP) ou negacao indevida (FN)? Com base nisso,
       o modelo escolhido no Exercicio 2c continua sendo o melhor para o
       banco? Que metrica voce olharia (precisao ou recall)?
    R:

Ex.4 - Qual modelo mais sofreu sem padronizacao e qual nao mudou nada?
       Explique o porque nos dois casos usando a ideia de DISTANCIA
       (k-NN, SVM) versus LIMIAR (arvore).
    R:

Ex.5 - O campeao foi sempre o mesmo nos 5 sorteios? O que isso diz sobre
       confiar em uma unica divisao treino/teste para declarar "o melhor
       modelo"? O que voce faria para ter mais confianca na comparacao?
    R:

=======================================================================
"""
