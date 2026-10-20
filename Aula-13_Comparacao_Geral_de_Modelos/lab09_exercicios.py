"""
Aprendizado de Maquina - IFSP Presidente Epitacio
Aula 9 - MINI-LABORATORIO  ***ARQUIVO DO ALUNO***
Comparacao honesta de todos os modelos supervisionados

"""
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, KFold, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_score,
                                     cross_validate, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")
SEMENTE = 42
pd.set_option("display.width", 110)


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


# ----- dados de REGRESSAO: anuncios de aluguel (mesmo arquivo das aulas de regressao)
alugueis = pd.read_csv("alugueis_tratado.csv")
Xa = alugueis[["area_m2", "quartos", "distancia_km"]]
ya = alugueis["valor_aluguel"]

# ----- dados de CLASSIFICACAO: solicitacoes de emprestimo
df = gerar_dados_emprestimo(n=450, semente=SEMENTE)
X = df.drop(columns="aprovado")
y = df["aprovado"]


# ============================================================ EXERCICIO 1 (2,0)
secao(1, "REGRESSAO: TODOS OS MODELOS COM 10 DOBRAS")

# TODO 1a: crie dobras10 = KFold(n_splits=10, shuffle=True, random_state=SEMENTE)
dobras10 = KFold(n_splits=10, shuffle=True, random_state=SEMENTE)
modelos_reg = {
    "Baseline (media)": DummyRegressor(strategy="mean"),
    "Linear multipla": LinearRegression(),
    "Polinomial grau 2": make_pipeline(PolynomialFeatures(2, include_bias=False), LinearRegression()),
    "Polinomial grau 6": make_pipeline(PolynomialFeatures(6, include_bias=False), StandardScaler(), LinearRegression()),
    "k-NN (k=15)": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15)),
    "Arvore (max_depth=5)": DecisionTreeRegressor(max_depth=5, random_state=SEMENTE),
    "SVR (C=1000, eps=50)": make_pipeline(StandardScaler(), SVR(C=1000, epsilon=50))
}

# TODO 1b: para cada modelo, rode cross_validate com cv=dobras10,
#          return_train_score=True e as tres metricas:
# TODO 1c: monte um DataFrame com essas colunas, ordene pelo RMSE (menor
#          primeiro) e imprima arredondado em 3 casas
linhas = []
for nome, modelo in modelos_reg.items():
    res = cross_validate(modelo, Xa, ya, cv=dobras10, return_train_score=True,
                         scoring={"MAE": "neg_mean_absolute_error",
                                  "RMSE": "neg_root_mean_squared_error", "R2": "r2"})
    mae = -res["test_MAE"]
    rmse = -res["test_RMSE"]
    r2 = res["test_R2"]
    rmse_treino = -res["train_RMSE"]
    linhas.append({
        "modelo": nome, "MAE": mae.mean(), "RMSE": rmse.mean(),
        "RMSE_desvio": rmse.std(), "R2": r2.mean(), "RMSE_treino": rmse_treino.mean()
    })

tab_reg = pd.DataFrame(linhas).set_index("modelo").sort_values("RMSE")
print(tab_reg.round(3))


# ============================================================ EXERCICIO 2 (1,5)
secao(2, "ARMADILHA: ARQUIVO ORDENADO E DOBRAS SEM EMBARALHAR")

# TODO 2a: crie uma copia de alugueis ORDENADA
ordenado = alugueis.sort_values("valor_aluguel").reset_index(drop=True)
Xo = ordenado[["area_m2", "quartos", "distancia_km"]]
yo = ordenado["valor_aluguel"]
knn_reg = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15))

# TODO 2b: calcule o R2 por dobra
sem_emb = cross_val_score(knn_reg, Xo, yo, cv=KFold(n_splits=5), scoring="r2")
com_emb = cross_val_score(knn_reg, Xo, yo, cv=KFold(n_splits=5, shuffle=True, random_state=SEMENTE), scoring="r2")
print(f"SEM shuffle - R2 por dobra: {sem_emb.round(3)}  media: {sem_emb.mean():.3f}")
print(f"COM shuffle - R2 por dobra: {com_emb.round(3)}  media: {com_emb.mean():.3f}")

# TODO 2c: imprimir min e max de cada dobra SEM shuffle
print("faixa de valor_aluguel em cada dobra SEM shuffle:")
for i, (idx_tr, idx_val) in enumerate(KFold(n_splits=5).split(Xo), start=1):
    val_aluguel = yo.iloc[idx_val]
    print(f"  dobra {i}: R$ {val_aluguel.min():.2f} a R$ {val_aluguel.max():.2f}")


# ============================================================ EXERCICIO 3 (2,0)
secao(3, "CLASSIFICACAO: SEIS FAMILIAS, 5 DOBRAS x 3 REPETICOES")

# TODO 3a: crie dobras_rep = RepeatedStratifiedKFold(n_splits=5, n_repeats=3,
#          random_state=SEMENTE) e o dicionario modelos_clf com:
#   "Baseline"              DummyClassifier(strategy="most_frequent")
#   "k-NN (k=5)"            KNeighborsClassifier(n_neighbors=5)
#   "Regressao logistica"   LogisticRegression(max_iter=1000)
#   "Arvore (max_depth=4)"  DecisionTreeClassifier(max_depth=4, random_state=SEMENTE)
#   "Random Forest"         RandomForestClassifier(n_estimators=200, random_state=SEMENTE)
#   "Naive Bayes"           GaussianNB()
#   "SVM (rbf)"             SVC(kernel="rbf")
dobras_rep = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEMENTE)
modelos_clf = {
    "Baseline": DummyClassifier(strategy="most_frequent"),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Regressao logistica": LogisticRegression(max_iter=1000),
    "Arvore (max_depth=4)": DecisionTreeClassifier(max_depth=4, random_state=SEMENTE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEMENTE),
    "Naive Bayes": GaussianNB(),
    "SVM (rbf)": SVC(kernel="rbf")
}

# TODO 3b: para cada modelo, cross_validate(com_prep(modelo), X, y,
#          cv=dobras_rep, scoring=["accuracy", "f1", "recall"]).
#          Guarde as 15 acuracias de cada modelo (vai precisar no 3c) e
#          imprima uma tabela com acuracia media, desvio, F1 e recall,
#          ordenada pela acuracia

linhas_clf = []
notas_acc = {}
for nome, modelo in modelos_clf.items():
    res = cross_validate(com_prep(modelo), X, y, cv=dobras_rep, scoring=["accuracy", "f1", "recall"])
    notas_acc[nome] = res["test_accuracy"]
    linhas_clf.append({
        "modelo": nome, "acuracia": res["test_accuracy"].mean(), "desvio": res["test_accuracy"].std(),
        "F1": res["test_f1"].mean(), "recall": res["test_recall"].mean()
    })

tab_clf = pd.DataFrame(linhas_clf).set_index("modelo").sort_values("acuracia", ascending=False)
print(tab_clf.round(3))

# TODO 3c: comparacao PAREADA com a regressao logistica: para cada modelo
#          (exceto Baseline e a propria regressao logistica), calcule a
#          diferenca dobra a dobra (acuracia do modelo - acuracia da RL) e
#          imprima a diferenca media e em quantas das 15 avaliacoes o modelo
#          venceu, empatou e perdeu
print("\ncomparacao pareada com a regressao logistica (15 avaliacoes):")
acc_rl = notas_acc["Regressao logistica"]
for nome in modelos_clf:
    if nome in ["Baseline", "Regressao logistica"]:
        continue
    diff = notas_acc[nome] - acc_rl
    venceu = (diff > 0).sum()
    empatou = (diff == 0).sum()
    perdeu = (diff < 0).sum()
    print(f"  {nome:<25} dif. media = {diff.mean():+.3f}   venceu {venceu:2}  empatou {empatou:2}  perdeu {perdeu:2}")


# ============================================================ EXERCICIO 4 (2,5)
secao(4, "AJUSTE JUSTO E VALIDACAO CRUZADA ANINHADA")

# TODO 4a: crie as dobras
interna = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)
externa = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE + 1)

grades = {
    "k-NN": (KNeighborsClassifier(n_neighbors=5), {"modelo__n_neighbors": [1, 5, 15, 45, 125]}),
    "Arvore": (DecisionTreeClassifier(max_depth=4, random_state=SEMENTE), {"modelo__max_depth": [2, 3, 4, 6, None]}),
    "SVM (rbf)": (SVC(kernel="rbf"), {"modelo__C": [0.1, 1, 10, 100], "modelo__gamma": ["scale", 0.01]})
}

# TODO 4b: para cada modelo, calcule a acuracia media da configuracao "de
#          fabrica" com cross_val_score(com_prep(modelo), X, y, cv=externa)

# TODO 4c: rode GridSearchCV(com_prep(modelo), grade, cv=interna).fit(X, y)
#          e guarde best_params_ e best_score_

# TODO 4d: calcule a validacao cruzada ANINHADA:
#          cross_val_score(GridSearchCV(com_prep(modelo), grade, cv=interna),
#                          X, y, cv=externa)
#          e imprima uma tabela com: config_padrao, melhor_params,
#          best_score_, aninhada (media) e otimismo = best_score_ - aninhada
linhas_ajuste = []
for nome, (modelo, grade) in grades.items():
    # modelo padrao
    padrao = cross_val_score(com_prep(modelo), X, y, cv=externa).mean()
    
    # GridSearch nas dobras internas com todo o dataset (fit normal, nao a aninhada)
    busca = GridSearchCV(com_prep(modelo), grade, cv=interna)
    busca.fit(X, y)
    
    # Remover o prefixo 'modelo__' para ficar igual ao gabarito
    params = {k.split('__')[1]: v for k, v in busca.best_params_.items()}
    melhor_params = str(params)
    
    # CV aninhada
    aninhada = cross_val_score(GridSearchCV(com_prep(modelo), grade, cv=interna), X, y, cv=externa).mean()
    
    linhas_ajuste.append({
        "modelo": nome, "config_padrao": padrao, "melhor_params": melhor_params,
        "best_score_": busca.best_score_, "aninhada": aninhada, 
        "otimismo": busca.best_score_ - aninhada
    })

tab_ajuste = pd.DataFrame(linhas_ajuste).set_index("modelo")
print(tab_ajuste.round(3).to_string())


# ============================================================ EXERCICIO 5 (2,0)
secao(5, "PROTOCOLO COMPLETO NA REGRESSAO: ESCOLHER NO TREINO, TESTAR UMA VEZ")

# TODO 5a: separe os alugueis em treino e teste com
#          train_test_split(Xa, ya, test_size=0.25, random_state=SEMENTE)
#          e crie cv_tr = KFold(n_splits=5, shuffle=True, random_state=SEMENTE)
#          A PARTIR DAQUI, O TESTE SO PODE SER USADO NO TODO 5c.
Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(Xa, ya, test_size=0.25, random_state=SEMENTE)
cv_tr = KFold(n_splits=5, shuffle=True, random_state=SEMENTE)

# TODO 5b: rode tres GridSearchCV no TREINO, todos com cv=cv_tr e
#          scoring="neg_root_mean_squared_error":
#   Polinomial: make_pipeline(PolynomialFeatures(include_bias=False), LinearRegression())
#               {"polynomialfeatures__degree": [1, 2, 3, 4]}
#   k-NN:       make_pipeline(StandardScaler(), KNeighborsRegressor())
#               {"kneighborsregressor__n_neighbors": [5, 10, 15, 25, 50]}
#   SVR:        make_pipeline(StandardScaler(), SVR(epsilon=50))
#               {"svr__C": [10, 100, 1000, 3000]}
#          Imprima o melhor parametro e o RMSE da CV (-best_score_) de cada um
grids_reg = {
    "Polinomial": (make_pipeline(PolynomialFeatures(include_bias=False), LinearRegression()), {"polynomialfeatures__degree": [1, 2, 3, 4]}),
    "k-NN": (make_pipeline(StandardScaler(), KNeighborsRegressor()), {"kneighborsregressor__n_neighbors": [5, 10, 15, 25, 50]}),
    "SVR": (make_pipeline(StandardScaler(), SVR(epsilon=50)), {"svr__C": [10, 100, 1000, 3000]})
}

melhor_rmse_cv = float("inf")
melhor_modelo_nome = None
modelo_final = None

for nome, (modelo, grade) in grids_reg.items():
    busca = GridSearchCV(modelo, grade, cv=cv_tr, scoring="neg_root_mean_squared_error")
    busca.fit(Xa_tr, ya_tr)
    rmse_cv = -busca.best_score_
    print(f"{nome:<12} melhor = {busca.best_params_}  RMSE na CV do treino = {rmse_cv:.2f}")
    if rmse_cv < melhor_rmse_cv:
        melhor_rmse_cv = rmse_cv
        melhor_modelo_nome = nome
        modelo_final = busca.best_estimator_

# TODO 5c: escolha o modelo com MENOR RMSE na CV do treino, pegue o
#          best_estimator_ dele (ja re-treinado com todo o treino), preveja
#          o TESTE uma unica vez e imprima MAE, RMSE e R2 no teste
print(f"\nescolhido: {melhor_modelo_nome} (RMSE estimado pela CV = {melhor_rmse_cv:.2f})")
ya_prev = modelo_final.predict(Xa_te)
rmse_teste = np.sqrt(mean_squared_error(ya_te, ya_prev))
print(f"TESTE (uma unica vez): MAE = {mean_absolute_error(ya_te, ya_prev):.2f} | RMSE = {rmse_teste:.2f} | R2 = {r2_score(ya_te, ya_prev):.3f}")





