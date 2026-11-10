"""
Aprendizado de Maquina - IFSP Presidente Epitacio
Laboratorio - Perceptron: treinamento, limites e comparacao com o Adaline
***ARQUIVO DO ALUNO***

NOME: ______________________________  PRONTUARIO: ______________  DATA: __/__/____

INSTRUCOES
  - Preencha os trechos marcados com  # TODO  (escreva o codigo logo abaixo de cada um)
  - Use SEMPRE a SEMENTE (42) para qualquer sorteio
  - O perceptron e o Adaline devem ser SEUS, em NumPy. Nao use o Perceptron do
    scikit-learn para treinar: ele so aparece no exercicio 7, como conferencia
  - Os exercicios usam a classe Perceptron que voce escrever no exercicio 1:
    confira-a bem antes de seguir
  - Rode o arquivo ate ele executar sem erro:   python lab_perceptron_exercicios.py
  - Responda as perguntas de interpretacao no bloco RELATORIO, no fim do arquivo
  - Entregue este arquivo executado, junto com a figura ex4_fronteira.png

VALE 10,0 PONTOS
  codigo 40%  |  relatorio 60%
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron as PerceptronSklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEMENTE = 42
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def secao(n, titulo):
    print()
    print("=" * 70)
    print(f"EXERCICIO {n} - {titulo}")
    print("=" * 70)


# ---------- dados usados nos exercicios (prontos, nao altere) ----------
X_PORTAS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
PORTAS = {"AND": [0, 0, 0, 1], "OR": [0, 1, 1, 1], "NAND": [1, 1, 1, 0],
          "NOR": [1, 0, 0, 0], "XOR": [0, 1, 1, 0]}


def dados_separaveis(n=100, semente=SEMENTE):
    """Duas classes separaveis pela reta x1 = x2, com uma faixa vazia (margem) em volta dela."""
    rng = np.random.default_rng(semente)
    X = rng.normal(size=(1500, 2)) * 1.5 + np.array([1.5, 1.5])
    dist = (X[:, 0] - X[:, 1]) / np.sqrt(2)          # distancia com sinal ate a reta x1 = x2
    X = X[np.abs(dist) > 0.4][:n]                    # descarta a faixa perto da reta: margem >= 0,4
    t = (X[:, 0] - X[:, 1] > 0).astype(int)
    return X, t


def dados_sobrepostos(semente=SEMENTE):
    """Duas classes que se sobrepoem: NAO sao linearmente separaveis."""
    X, t = make_blobs(n_samples=300, centers=[(-1, -1), (1, 1)], cluster_std=1.6, random_state=semente)
    return train_test_split(X, t, test_size=0.3, random_state=semente, stratify=t)


# ============================================================ EXERCICIO 1 (1,5)
secao(1, "O PERCEPTRON EM NUMPY")

# TODO 1a: complete a classe Perceptron abaixo. O __init__ ja esta pronto. Escreva:
#            fit(X, t)   -> treina com a regra do perceptron (t em {0, 1}).
#                           Comece com w = w0 (zeros se w0 for None) e b = b0.
#                           A cada epoca percorra os exemplos (em ordem fixa, ou
#                           embaralhados com rng.permutation se embaralhar=True),
#                           corrija so quando errar e pare ao fechar uma epoca
#                           sem erros ou ao atingir max_epocas. Guarde:
#                             self.erros_     lista com o no de erros de CADA epoca
#                             self.historico_ lista de (w, b) ao FIM de cada epoca
#                             self.convergiu_ True se a ultima epoca teve 0 erros
#            z(X)        -> X @ w + b      (X: vetor (d,) ou lote (n, d))
#            predict(X)  -> 1 onde z >= 0, 0 caso contrario
#            score(X, t) -> acuracia
class Perceptron:
    def __init__(self, eta=1.0, max_epocas=100, embaralhar=False, semente=SEMENTE, w0=None, b0=0.0):
        self.eta, self.max_epocas, self.embaralhar, self.semente = eta, max_epocas, embaralhar, semente
        self.w0, self.b0 = w0, b0

    # escreva aqui os metodos z, predict, score e fit (dentro da classe)



# TODO 1b: treine o AND (X_PORTAS, PORTAS["AND"]) com eta = 1, w e b iniciais nulos
#          e ordem fixa. Imprima os erros por epoca, w e b. Confira com o exemplo
#          feito a mao na apostila: erros 2, 3, 3, 2, 1, 0; w = (2; 1); b = -3.
#          Use assert para falhar se algum valor divergir.


# TODO 1c: faca o mesmo para OR e NOR e confira com os Exercicios 1 e 6 da apostila
#          de conceitos. OR: erros 2, 2, 1, 0; w = (1; 1); b = -1.
#          NOR: erros 1, 2, 1, 0; w = (-1; -1); b = 0.



# ============================================================ EXERCICIO 2 (1,5)
secao(2, "PORTAS LOGICAS E O XOR")

# TODO 2a: treine AND, OR, NAND, NOR e XOR (eta = 1, ordem fixa, max_epocas = 50).
#          Monte um DataFrame com uma linha por porta e as colunas: porta,
#          convergiu, epocas (len(erros_)), w e b finais. Imprima-o.


# TODO 2b: no XOR, imprima os erros das 10 primeiras epocas. Depois conte quantos
#          pares (w, b) DISTINTOS aparecem nos fins de epoca de 3 a 50
#          (use modelos["XOR"].historico_ e um set de tuplas).


# TODO 2c: treine o AND com eta em {1, 0.5, 0.25, 0.1} (w e b iniciais nulos, ordem
#          fixa). Para cada eta imprima os erros por epoca e o w e b finais. Para
#          cada eta confira com np.allclose(pesos_eta / eta, pesos_eta1), onde "pesos" e o
#          vetor (w1, w2, b), se os pesos finais sao proporcionais a eta, e se a
#          sequencia de erros e a mesma de eta = 1. Observe o resultado com eta = 0.1.



# ============================================================ EXERCICIO 3 (1,5)
secao(3, "DADOS SEPARAVEIS: LIMITE DE ERROS E TAXA DE APRENDIZADO")
X3, t3 = dados_separaveis()

# TODO 3a: treine um Perceptron (eta = 1, ordem fixa) em X3, t3. Imprima o numero
#          de exemplos, os erros por epoca, o total de correcoes (soma dos erros),
#          w, b e a acuracia no treino.


# TODO 3b: confira o teorema de convergencia. Os dados sao separados pela reta
#          x1 = x2, isto e, por w* = (1; -1; 0) / raiz(2) no espaco aumentado (x, 1).
#          Calcule a margem gamma = menor valor de t'*(w* . x_aum), com t' = 2t-1,
#          e R = maior norma de x_aum = (x1, x2, 1). Imprima gamma, R, o limite
#          (R / gamma)**2 e o total de correcoes do item 3a. Use assert para
#          garantir total <= limite.


# TODO 3c: (i) com w e b iniciais nulos, treine com eta em {1, 0.5, 0.25} e confira
#          que os erros por epoca sao identicos e que w / eta e o mesmo.
#          (ii) com w0 = (-1.0; 0.5) e b0 = 0.3, treine com eta = 1 e com eta = 0.25
#          e imprima os erros por epoca de cada um: eles sao iguais agora?



# ============================================================ EXERCICIO 4 (1,0)
secao(4, "A FRONTEIRA SE MOVE")

# TODO 4a: com o modelo m3 do exercicio 3, faca uma figura com tres paineis lado a lado:
#          fronteira ao fim da epoca 1, da epoca 2 e da ultima epoca (use m3.historico_).
#          A fronteira e a reta w1*x1 + w2*x2 + b = 0. Desenhe os pontos de cada classe
#          com cores diferentes, escreva no titulo a epoca e o numero de erros dela,
#          e salve como "ex4_fronteira.png".



# ============================================================ EXERCICIO 5 (1,5)
secao(5, "DADOS NAO SEPARAVEIS E O ALGORITMO DO BOLSO")
X5tr, X5te, t5tr, t5te = dados_sobrepostos()

# TODO 5a: treine um Perceptron (eta = 1, embaralhar=True, max_epocas = 100) em X5tr, t5tr.
#          Imprima: convergiu_?, os erros das 5 ultimas epocas, o MENOR no de erros
#          entre todas as epocas, e a acuracia no treino e no teste dos pesos FINAIS.


# TODO 5b: escreva perceptron_bolso(X, t, eta, max_epocas, semente) que treina como o
#          perceptron (embaralhando a cada epoca), mas depois de CADA correcao calcula
#          os erros do (w, b) atual em TODO o conjunto de treino e guarda no "bolso"
#          o melhor par visto ate agora. Devolve (w_bolso, b_bolso, erros_bolso).


# TODO 5c: rode perceptron_bolso em X5tr, t5tr. Imprima o menor numero de erros no
#          treino e a acuracia no treino e no teste do par do bolso. Compare com os
#          pesos finais (5a) e com uma LogisticRegression (referencia) no teste.



# ============================================================ EXERCICIO 6 (1,5)
secao(6, "ADALINE: A REGRA DELTA")

# TODO 6a: complete a classe Adaline abaixo (o __init__ ja esta pronto) com fit, z, predict
#          e score. Regra delta: para cada exemplo, e = t - z_i e
#          w <- w + eta * e * x_i, b <- b + eta * e (t em {0, 1}, sem degrau no erro).
#          Embaralhe a cada epoca e guarde self.mse_ = erro quadratico medio de TODO o
#          conjunto ao fim de cada epoca. predict devolve 1 onde z >= 0.5.
class Adaline:
    def __init__(self, eta=0.01, max_epocas=50, embaralhar=True, semente=SEMENTE):
        self.eta, self.max_epocas, self.embaralhar, self.semente = eta, max_epocas, embaralhar, semente

    # escreva aqui os metodos z, predict, score e fit (dentro da classe)


# TODO 6b: padronize X5tr e X5te com StandardScaler (ajuste SO no treino). Treine o Adaline
#          (30 epocas) com eta em {0.001, 0.01, 0.1, 0.5, 1.0}. Monte uma tabela com eta,
#          MSE da epoca 1, MSE final e acuracia no teste. Se o MSE nao for um numero finito,
#          escreva "divergiu". Use np.errstate(all="ignore") para calar avisos de overflow.


# TODO 6c: calcule o limite de estabilidade 2 / max(||x||^2 + 1) sobre A5tr (o fator da
#          Secao 7 da apostila, tomado no pior exemplo) e imprima-o. Depois compare, no
#          teste, o Adaline com eta = 0.01 e o perceptron do bolso do exercicio 5
#          (treinado nos dados padronizados A5tr).



# ============================================================ EXERCICIO 7 (1,5)
secao(7, "A IRIS: UM PAR SEPARAVEL E UM NAO SEPARAVEL")
iris = load_iris()


def par_iris(c0, c1):
    m = np.isin(iris.target, [c0, c1])
    X, t = iris.data[m], (iris.target[m] == c1).astype(int)
    Xtr, Xte, ttr, tte = train_test_split(X, t, test_size=0.3, random_state=SEMENTE, stratify=t)
    esc = StandardScaler().fit(Xtr)
    return esc.transform(Xtr), esc.transform(Xte), ttr, tte


# TODO 7a: setosa (0) contra versicolor (1) com par_iris(0, 1) (4 atributos, ja padronizados).
#          Treine um Perceptron (embaralhar=True, max_epocas = 200). Imprima convergiu_, o
#          numero de epocas, a acuracia no treino e no teste e a matriz de confusao no teste
#          (sklearn.metrics.confusion_matrix). Confira a acuracia de teste com
#          sklearn.linear_model.Perceptron (eta0=1.0, tol=None, max_iter=200, random_state=42).


# TODO 7b: versicolor (1) contra virginica (2) com par_iris(1, 2). Treine (i) o Perceptron
#          comum (embaralhar=True, 200 epocas) e (ii) o perceptron do bolso do exercicio 5.
#          Imprima convergiu_, os erros das 5 ultimas epocas e a acuracia de teste de cada
#          um, a matriz de confusao do bolso e a acuracia de teste do sklearn Perceptron.


# TODO 7c: treine o Adaline (eta = 0.01, 30 epocas) no par versicolor x virginica e imprima
#          a acuracia de teste e o MSE final. Compare com os resultados de 7b.



# ============================================================ RELATORIO
"""
=======================================================================
RELATORIO  (vale 60% da nota desta sessao)
=======================================================================

Ex.1 - (a) Por que a regra so altera w e b quando o perceptron erra? (b) O
       que significam, no exemplo do AND, os valores 2, 3, 3, 2, 1, 0 de
       erros por epoca? Eles diminuem a cada epoca? (c) Por que ordem fixa e
       pesos iniciais nulos tornam o resultado reproduzivel e comparavel
       com o feito a mao?
    R:

Ex.2 - (a) O que a tabela mostra sobre AND, OR, NAND e NOR contra o XOR? (b)
       Quantos pares (w, b) distintos aparecem nos fins de epoca de 3 a 50
       no XOR e o que isso diz sobre o treino? (c) O que muda no AND com
       eta = 0.1, comparado a eta = 1, 0.5 e 0.25? Por que?
    R:

Ex.3 - (a) Quantas correcoes o perceptron fez e em quantas epocas? (b) O
       limite do teorema e muito maior que o numero real de correcoes: isso
       e uma falha do teorema? (c) O que o item 3c mostra sobre o papel de
       eta com w0 = 0 e com w0 diferente de zero?
    R:

Ex.4 - (a) Descreva como a fronteira se move da epoca 1 a ultima. (b) A
       fronteira final e a unica que separa os dados? Por que a final
       difere da reta x1 = x2 que gerou as classes?
    R:

Ex.5 - (a) O perceptron convergiu? O que o comportamento dos erros mostra?
       (b) O que o bolso guarda e por que ele ajuda? (c) A acuracia do
       bolso no teste e maior que a dos pesos finais? Isso e garantido?
    R:

Ex.6 - (a) Qual a diferenca, no codigo, entre a regra do perceptron e a regra
       delta? (b) O que a tabela mostra sobre eta = 0.5 e 1.0 e qual a
       relacao com o limite do item 6c? (c) Adaline ou perceptron: qual foi
       melhor no teste e por que o Adaline nao precisa de separabilidade?
    R:

Ex.7 - (a) Por que setosa x versicolor converge e versicolor x virginica nao?
       (b) O que a matriz de confusao do bolso mostra? (c) Por que a conferencia
       com o sklearn nao precisa dar exatamente o mesmo resultado?
    R:

=======================================================================
"""

print("""
=======================================================================
Arquivo executado. Confira se todos os TODO foram preenchidos e se o
bloco RELATORIO esta respondido antes de entregar.
=======================================================================""")
