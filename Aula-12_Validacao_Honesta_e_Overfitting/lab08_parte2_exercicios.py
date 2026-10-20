"""
Aprendizado de Maquina - IFSP Presidente Epitacio
Aula 8, Parte 2 - MINI-LABORATORIO  ***ARQUIVO DO ALUNO***

NOME: ______________________________  PRONTUARIO: ______________  DATA: __/__/____

INSTRUCOES
  - Preencha os trechos marcados com  # TODO
  - Use SEMPRE random_state=42 (SEMENTE)
  - As funcoes gerar_dados_emprestimo(), montar_preprocessador() e
    com_prep() ja estao prontas abaixo - nao precisa alterar
  - Rode o arquivo ate ele executar sem erro:   python lab08_parte2_exercicios.py
  - Responda as perguntas de interpretacao no bloco RELATORIO, no fim do arquivo
  - Entregue este arquivo executado

VALE 10,0 PONTOS
  codigo 30%  |  relatorio 70%  (sessao mais conceitual que a anterior)
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split, validation_curve)
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


def montar_preprocessador():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("esc", StandardScaler())]), COL_NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), COL_CAT),
    ])


def com_prep(modelo):
    return Pipeline([("prep", montar_preprocessador()), ("modelo", modelo)])


df = gerar_dados_emprestimo(n=450, semente=SEMENTE)
X = df.drop(columns="aprovado")
y = df["aprovado"]

# ============================================================ EXERCICIO 1 (1,5)
secao(1, "DIAGNOSTICO POR CENARIO (sem codigo)")

# TODO 1: sem codigo. Cada cenario abaixo descreve um PROBLEMA de avaliacao.
#         Diga no RELATORIO qual e o problema (sobreajuste, subajuste,
#         vazamento de dados, acuracia enganosa por desbalanceamento ou
#         reuso do teste) e o que voce faria para corrigir.
#   (a) Acuracia de 99% no treino e de 71% na validacao.
#   (b) Acuracia de 62% no treino e de 60% na validacao, e o baseline
#       "sempre a classe mais comum" ja faz 58%.
#   (c) O analista padroniza o dataset INTEIRO com StandardScaler e so
#       depois separa treino e teste.
#   (d) Um modelo de deteccao de fraude tem 99% de acuracia, mas fraudes
#       sao 1% dos exemplos e o modelo nao detecta nenhuma.
#   (e) O analista testou 30 configuracoes diferentes, olhando sempre a
#       acuracia no conjunto de teste, e reporta a melhor delas.

# ============================================================ EXERCICIO 2 (2,5)
secao(2, "A CURVA DE COMPLEXIDADE DO K-NN")

# TODO 2a: com validation_curve, calcule a acuracia de TREINO e de VALIDACAO
#          (media das 5 dobras de StratifiedKFold(n_splits=5, shuffle=True,
#          random_state=SEMENTE)) do k-NN com estes valores de k:
#          [1, 3, 5, 9, 15, 25, 45, 75, 125, 200, 300]
#          Dica: validation_curve(com_prep(KNeighborsClassifier()), X, y,
#                param_name="modelo__n_neighbors", param_range=ks, cv=...)

# TODO 2b: monte uma tabela com as colunas k, treino, validacao e lacuna
#          (lacuna = treino - validacao) e imprima arredondada em 3 casas

# TODO 2c: imprima o k com a MAIOR acuracia de validacao

k = [1, 3, 5, 9, 15, 25, 45, 75, 125, 200, 300]
tr_k, va_k = validation_curve(com_prep(KNeighborsClassifier()), X, y,
                              param_name="modelo__n_neighbors", param_range=k, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE))

print(pd.DataFrame({"k": k, "treino": tr_k.mean(1), "validação": va_k.mean(1), "lacuna": (tr_k.mean(1)-va_k.mean(1))}).round(3))

maior_acc = -2
maior_k = 0

for k_atual, validacao in zip(k, va_k.mean(1)):
    if validacao > maior_acc:
        maior_acc = validacao
        maior_k = k_atual

print(f"\nO melhor K: {maior_k} | Acurácia do melhor K: {maior_acc:.3f}")

# ============================================================ EXERCICIO 3 (2,0)
secao(3, "COMPARACAO PAREADA COM 10 DOBRAS")

# TODO 3a: crie dobras10 = StratifiedKFold(n_splits=10, shuffle=True,
#          random_state=SEMENTE) e calcule, com cross_val_score e o MESMO
#          objeto dobras10, a acuracia por dobra de:
#          Regressao Logistica (max_iter=1000), SVM (kernel="rbf") e
#          Arvore de Decisao (max_depth=4). Use com_prep(...) em todos.

dobras10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEMENTE)

notas_rl = cross_val_score(com_prep(LogisticRegression(max_iter=1000)), X, y, cv=dobras10)
notas_svm = cross_val_score(com_prep(SVC(kernel="rbf", random_state=SEMENTE)), X, y, cv=dobras10)
notas_arv = cross_val_score(com_prep(DecisionTreeClassifier(max_depth=4, random_state=SEMENTE)), X, y, cv=dobras10)

# TODO 3b: imprima media e desvio de cada um
print(f"Regressao Logistica: media={notas_rl.mean():.3f}, desvio={notas_rl.std():.3f}")
print(f"SVM: media={notas_svm.mean():.3f}, desvio={notas_svm.std():.3f}")
print(f"Arvore de Decisao: media={notas_arv.mean():.3f}, desvio={notas_arv.std():.3f}")

# TODO 3c: para os pares (Regressao Logistica - Arvore) e (SVM - Regressao
#          Logistica), calcule a diferenca dobra a dobra e imprima a
#          diferenca media e quantas dobras cada lado venceu, empatou e perdeu

diff_rl_arv = notas_rl - notas_arv
vit_rl_arv = (diff_rl_arv > 0).sum()
emp_rl_arv = (diff_rl_arv == 0).sum()
der_rl_arv = (diff_rl_arv < 0).sum()

print(f"\nRL vs Arvore: diff media={diff_rl_arv.mean():.3f} | RL venceu {vit_rl_arv}, Arvore venceu {der_rl_arv}, empates {emp_rl_arv}")

diff_svm_rl = notas_svm - notas_rl
vit_svm_rl = (diff_svm_rl > 0).sum()
emp_svm_rl = (diff_svm_rl == 0).sum()
der_svm_rl = (diff_svm_rl < 0).sum()

print(f"SVM vs RL: diff media={diff_svm_rl.mean():.3f} | SVM venceu {vit_svm_rl}, RL venceu {der_svm_rl}, empates {emp_svm_rl}")

# ============================================================ EXERCICIO 4 (2,0)
secao(4, "CLASSES DESBALANCEADAS E CUSTO DO ERRO")

# TODO 4a: gere o problema desbalanceado:
X_r, y_r = make_classification(n_samples=1000, n_features=6,
    n_informative=4, n_redundant=0, n_clusters_per_class=1,
    weights=[0.90, 0.10], flip_y=0.0, class_sep=1.0, random_state=SEMENTE)

X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.30, random_state=SEMENTE, stratify=y_r)

# TODO 4b: treine (1) DummyClassifier(strategy="most_frequent"),
#          (2) LogisticRegression(max_iter=1000) e
#          (3) LogisticRegression(max_iter=1000, class_weight="balanced")
candidatos = {
    "Dummy": DummyClassifier(strategy="most_frequent"),
    "LogReg": LogisticRegression(max_iter=1000),
    "LogReg_Balanced": LogisticRegression(max_iter=1000, class_weight="balanced"),
}

# TODO 4c: para cada um, imprima acuracia, precisao, recall, F1 e os
#          numeros FN (fraudes perdidas) e FP (alarmes falsos)
# TODO 4d: suponha que perder uma fraude (FN) custa 10 vezes mais que um
#          alarme falso (FP). Calcule custo = 10*FN + 1*FP de cada modelo
#          e imprima qual tem o menor custo
linhas = []
for nome, modelo in candidatos.items():
    modelo.fit(X_tr, y_tr)
    y_prev = modelo.predict(X_te)
    tn, fp, fn, tp = confusion_matrix(y_te, y_prev).ravel()
    custo = 10 * fn + 1 * fp
    linhas.append((nome, accuracy_score(y_te, y_prev), precision_score(y_te, y_prev, zero_division=0),
                   recall_score(y_te, y_prev), f1_score(y_te, y_prev), fn, fp, custo))

res = pd.DataFrame(linhas, columns=["modelo", "acuracia", "precisao", "recall", "f1",
                                     "FN", "FP", "custo"]).set_index("modelo").round(3)
print(res)
print(f"\nModelo com menor custo: {res['custo'].idxmin()} (Custo = {res['custo'].min()})")

# ============================================================ EXERCICIO 5 (2,0)
secao(5, "VAZAMENTO DE DADOS")

# TODO 5a: repita 20 vezes (semente em range(20)): gere
#          rng = np.random.default_rng(100 + semente),
#          X_ruido = rng.normal(size=(80, 1500)), y_ruido = rng.integers(0, 2, 80)
#          (puro ruido - nao ha padrao nenhum)
# TODO 5b: FORMA ERRADA - selecione os 10 melhores atributos com
#          SelectKBest(f_classif, k=10).fit_transform(X_ruido, y_ruido) usando
#          TODAS as linhas e depois avalie LogisticRegression(max_iter=1000)
#          com cross_val_score (StratifiedKFold(5, shuffle=True, random_state=SEMENTE))
# TODO 5c: FORMA CERTA - coloque SelectKBest e LogisticRegression dentro de
#          um Pipeline e passe o Pipeline (com os dados ORIGINAIS X_ruido)
#          ao cross_val_score

errado, certo = [], []
for semente in range(20):
    rng = np.random.default_rng(100 + semente)
    X_ruido = rng.normal(size=(80, 1500))
    y_ruido = rng.integers(0, 2, 80)
    dobras = StratifiedKFold(5, shuffle=True, random_state=SEMENTE)

    # ERRADO
    X_sel = SelectKBest(f_classif, k=10).fit_transform(X_ruido, y_ruido)
    errado.append(cross_val_score(LogisticRegression(max_iter=1000), X_sel, y_ruido, cv=dobras).mean())

    # CERTO
    pipe = Pipeline([("sel", SelectKBest(f_classif, k=10)), ("modelo", LogisticRegression(max_iter=1000))])
    certo.append(cross_val_score(pipe, X_ruido, y_ruido, cv=dobras).mean())

# TODO 5d: imprima a acuracia media das duas formas nas 20 repeticoes
print(f"FORMA ERRADA: acuracia media = {np.mean(errado):.3f}")
print(f"FORMA CERTA: acuracia media = {np.mean(certo):.3f}")


# ============================================================ RELATORIO
"""
=======================================================================
RELATORIO  (vale 70% da nota desta sessao)
=======================================================================

Ex.1 - Para cada cenario (a)-(e): qual e o problema e como corrigir?
    R:

Ex.2 - Em que faixa de k esta o sobreajuste? E o subajuste? Qual k voce
       escolheria e por que NAO escolheria k = 1 mesmo com 100% no treino?
    R:

Ex.3 - Regressao Logistica vs Arvore: a diferenca media e clara? Olhe
       tambem quantas dobras cada uma venceu. E SVM vs Regressao
       Logistica: existe um vencedor? O que "empate" significa aqui?
    R:

Ex.4 - Por que o baseline tem acuracia alta e mesmo assim e inutil?
       Com o custo 10*FN + FP, qual modelo voce escolheria? Isso coincide
       com o modelo de maior acuracia? Justifique.
    R:

Ex.5 - Por que a forma errada reporta uma acuracia alta em dados de puro
       ruido? Onde exatamente "vazou" informacao? Como o Pipeline resolve?
    R:

=======================================================================
"""

print("""
=======================================================================
Arquivo executado. Confira se todos os TODO foram preenchidos e se o
bloco RELATORIO esta respondido antes de entregar.
=======================================================================""")
