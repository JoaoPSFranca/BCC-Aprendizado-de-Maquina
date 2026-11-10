"""
Aprendizado de Maquina - IFSP Presidente Epitacio
Laboratorio - Implementacao de um neuronio artificial e comparacao de funcoes de ativacao
***ARQUIVO DO ALUNO***

NOME: João Pedro S. Franca  PRONTUARIO: pe3021114  DATA: 10/11/2026

INSTRUCOES
  - Preencha os trechos marcados com  # TODO
  - Use SEMPRE random_state=SEMENTE (42) e np.random.default_rng(SEMENTE)
  - Os exercicios usam as funcoes e a classe que voce escrever no exercicio 1:
    confira-as bem antes de seguir
  - Nao use bibliotecas de redes neurais (TensorFlow, PyTorch, MLPClassifier):
    o neuronio deve ser seu, em NumPy. O scikit-learn so aparece no exercicio 6
  - Rode o arquivo ate ele executar sem erro:   python lab_neuronio_exercicios.py
  - Responda as perguntas de interpretacao no bloco RELATORIO, no fim do arquivo
  - Entregue este arquivo executado

VALE 10,0 PONTOS
  codigo 40%  |  relatorio 60%
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEMENTE = 42
pd.set_option("display.width", 110)

def secao(n, titulo):
    print()
    print("=" * 70)
    print(f"EXERCICIO {n} - {titulo}")
    print("=" * 70)


# ============================================================ EXERCICIO 1 (1,5)
secao(1, "O NEURONIO EM NUMPY")

# TODO 1a: escreva as funcoes degrau, identidade, sigmoide, tanh, relu e
#          leaky_relu (alpha=0.01). Todas devem aceitar escalar OU vetor NumPy.
#          Depois monte o dicionario ATIVACOES = {"degrau": degrau, ...}
def degrau(z):
    return np.where(np.asarray(z, dtype=float) >= 0, 1.0, 0.0)

def identidade(z):
    return np.asarray(z, dtype=float)

def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))

def tanh(z):
    return np.tanh(np.asarray(z, dtype=float))

def relu(z):
    return np.maximum(0.0, np.asarray(z, dtype=float))

def leaky_relu(z, alpha=0.01):
    z = np.asarray(z, dtype=float)
    return np.where(z > 0, z, alpha * z)

ATIVACOES = {
    "degrau": degrau,
    "identidade": identidade,
    "sigmoide": sigmoide,
    "tanh": tanh,
    "relu": relu,
    "leaky_relu": leaky_relu,
}

# TODO 1b: escreva a classe Neuronio(w, b, ativacao) com os metodos
#            z(X)     -> soma ponderada + vies (X pode ser um vetor (d,) ou
#                        um lote (n, d); use X @ w + b)
#            saida(X) -> a ativacao (buscada em ATIVACOES pelo nome) aplicada a z
#          Confira com x = (3; 1; -2), w = (0,2; -0,5; 0,6), b = 0,1:
#          imprima z e a saida de CADA uma das seis ativacoes
class Neuronio:
    def __init__(self, w, b, ativacao="sigmoide"):
        self.w = np.asarray(w, dtype=float)
        self.b = float(b)
        self.ativacao = ativacao

    def z(self, X):
        return np.asarray(X, dtype=float) @ self.w + self.b

    def saida(self, X):
        return ATIVACOES[self.ativacao](self.z(X))

x_1b = np.array([3.0, 1.0, -2.0])
w_1b = np.array([0.2, -0.5, 0.6])
b_1b = 0.1
n_teste = Neuronio(w_1b, b_1b)
print(f"z = {n_teste.z(x_1b):+.3f}")
for nome in ATIVACOES:
    n_ativ = Neuronio(w_1b, b_1b, nome)
    print(f"  {nome:<11} y = {n_ativ.saida(x_1b):+.3f}")

# TODO 1c: com o lote abaixo e a ativacao tanh, calcule as saidas de uma vez
#          (lote inteiro) e por um laco, amostra por amostra. Imprima z, y e
#          o resultado de np.allclose(lote, laco)
lote = np.array([[3, 1, -2], [0, 0, 0], [-1, 2, 1], [4, -3, 0.5]], dtype=float)
n_tanh = Neuronio(w_1b, b_1b, "tanh")
z_lote = n_tanh.z(lote)
y_lote = n_tanh.saida(lote)
y_laco = np.array([n_tanh.saida(amostra) for amostra in lote])

print("z (lote):", z_lote)
print("y (lote):", y_lote)
print("np.allclose(lote, laco):", np.allclose(y_lote, y_laco))

# TODO 1d: confira sigmoide e tanh com os valores de referencia da apostila de
#          conceitos (use np.allclose com atol=5e-4 e imprima True/False)
zs_ref = np.array([-4, -2, -1, 0, 1, 2, 4], dtype=float)
REF_SIGMOIDE = np.array([0.018, 0.119, 0.269, 0.500, 0.731, 0.881, 0.982])
REF_TANH = np.array([-0.999, -0.964, -0.762, 0.000, 0.762, 0.964, 0.999])
confere_sig = np.allclose(sigmoide(zs_ref), REF_SIGMOIDE, atol=5e-4)
confere_tanh = np.allclose(tanh(zs_ref), REF_TANH, atol=5e-4)
print(f"Sigmoide confere com referencia: {confere_sig}")
print(f"Tanh confere com referencia:     {confere_tanh}")

# ============================================================ EXERCICIO 2 (1,5)
secao(2, "COMPARANDO AS ATIVACOES NAS MESMAS ENTRADAS")

# TODO 2a: com zs = np.linspace(-6, 6, 13), monte um DataFrame tab2 (indice z,
#          uma coluna por ativacao) e imprima com 3 casas decimais
zs = np.linspace(-6, 6, 13)
tab2 = pd.DataFrame({nome: ATIVACOES[nome](zs) for nome in ATIVACOES}, index=zs)
tab2.index.name = "z"
print(tab2.round(3).to_string())

# TODO 2b: escreva as derivadas d_sigmoide, d_tanh, d_relu, d_leaky_relu,
#          d_degrau (zero) e d_identidade (um) e monte DERIVADAS = {...}.
#          Confira as quatro primeiras contra a derivada NUMERICA
#          (f(z+h) - f(z-h)) / (2h), h = 1e-5, em z_teste (sem o z = 0, onde a
#          ReLU nao tem derivada). Imprima a maior diferenca de cada uma
z_teste = np.array([-3.0, -1.0, -0.5, 0.5, 1.0, 3.0])

def d_degrau(z):
    return np.zeros_like(np.asarray(z, dtype=float))

def d_identidade(z):
    return np.ones_like(np.asarray(z, dtype=float))

def d_sigmoide(z):
    s = sigmoide(z)
    return s * (1.0 - s)

def d_tanh(z):
    return 1.0 - np.tanh(np.asarray(z, dtype=float)) ** 2

def d_relu(z):
    return np.where(np.asarray(z, dtype=float) > 0, 1.0, 0.0)

def d_leaky_relu(z, alpha=0.01):
    return np.where(np.asarray(z, dtype=float) > 0, 1.0, alpha)

DERIVADAS = {
    "degrau": d_degrau,
    "identidade": d_identidade,
    "sigmoide": d_sigmoide,
    "tanh": d_tanh,
    "relu": d_relu,
    "leaky_relu": d_leaky_relu,
}

h = 1e-5
for nome in ["sigmoide", "tanh", "relu", "leaky_relu"]:
    f = ATIVACOES[nome]
    df_analitica = DERIVADAS[nome](z_teste)
    df_numerica = (f(z_teste + h) - f(z_teste - h)) / (2 * h)
    dif_max = np.max(np.abs(df_analitica - df_numerica))
    print(f"{nome:<11} maior diferenca contra numerica: {dif_max:.2e}")

# TODO 2c: com fina = np.linspace(-10, 10, 2001), monte uma tabela com, para
#          degrau, sigmoide, tanh, relu e leaky_relu: minimo, maximo e media
#          da saida, derivada maxima e a FRACAO de pontos com derivada < 0,01
#          (fracao "saturada"). Imprima com 3 casas
fina = np.linspace(-10, 10, 2001)
linhas_2c = []
for nome in ["degrau", "sigmoide", "tanh", "relu", "leaky_relu"]:
    y_fina = ATIVACOES[nome](fina)
    dy_fina = DERIVADAS[nome](fina)
    linhas_2c.append({
        "ativacao": nome,
        "minimo": np.min(y_fina),
        "maximo": np.max(y_fina),
        "media": np.mean(y_fina),
        "derivada_maxima": np.max(dy_fina),
        "fracao_saturada": np.mean(dy_fina < 0.01),
    })
tab_2c = pd.DataFrame(linhas_2c).set_index("ativacao")
print(tab_2c.round(3).to_string())

# ============================================================ EXERCICIO 3 (1,5)
secao(3, "PORTAS LOGICAS, BUSCA EM GRADE E ESCALA DOS PESOS")

# TODO 3a: encontre a mao w e b (neuronio de degrau) para tres portas:
#            NOT   (1 entrada: saida 1 quando a entrada e 0)
#            AND3  (3 entradas: 1 so quando as tres sao 1)
#            MAJ3  (3 entradas: 1 quando pelo menos DUAS sao 1)
#          Para cada uma, percorra itertools.product([0, 1], repeat=n), imprima
#          as saidas e confira com a regra em Python (all(...), sum(...) >= 2)
portas_3a = {
    "NOT": (1, [-1.0], 0.5, lambda x: int(not x[0])),
    "AND3": (3, [1.0, 1.0, 1.0], -2.5, lambda x: int(all(x))),
    "MAJ3": (3, [1.0, 1.0, 1.0], -1.5, lambda x: int(sum(x) >= 2)),
}
for nome, (n_in, w_p, b_p, regra) in portas_3a.items():
    neuronio = Neuronio(w_p, b_p, "degrau")
    entradas = list(itertools.product([0, 1], repeat=n_in))
    saidas = [int(neuronio.saida(x)) for x in entradas]
    esperadas = [regra(x) for x in entradas]
    print(f"{nome:<5} w={w_p} b={b_p:+} saidas={saidas} confere={saidas == esperadas}")

# TODO 3b: com grade = np.arange(-2, 2.01, 0.5), teste TODAS as combinacoes
#          (w1, w2, b) com degrau e conte quantas reproduzem AND, OR, XOR e
#          XNOR nas entradas (0,0), (0,1), (1,0), (1,1). Imprima as contagens
grade = np.arange(-2, 2.01, 0.5)
entradas2 = list(itertools.product([0, 1], repeat=2))
alvos_3b = {
    "AND": [0, 0, 0, 1],
    "OR": [0, 1, 1, 1],
    "XOR": [0, 1, 1, 0],
    "XNOR": [1, 0, 0, 1],
}
contagem_3b = {k: 0 for k in alvos_3b}
for w1, w2, b_val in itertools.product(grade, grade, grade):
    n = Neuronio([w1, w2], b_val, "degrau")
    s = [int(n.saida(e)) for e in entradas2]
    for k, v in alvos_3b.items():
        if s == v:
            contagem_3b[k] += 1

print(f"Combinacoes testadas: {len(grade) ** 3}")
for k, v in contagem_3b.items():
    print(f"  {k:<5} solucoes encontradas: {v:>3}")

# TODO 3c: porta AND com w = (1; 1), b = -1,5 e ativacao SIGMOIDE. Imprima as
#          quatro saidas com os pesos e o vies multiplicados por k = 1 e por
#          k = 10 (4 casas decimais)
for k_val in [1, 10]:
    n_sig = Neuronio(np.array([1.0, 1.0]) * k_val, -1.5 * k_val, "sigmoide")
    saidas_sig = [n_sig.saida(e) for e in entradas2]
    print(f"k = {k_val:>2}: {[round(float(val), 4) for val in saidas_sig]}")

# ============================================================ EXERCICIO 4 (1,5)
secao(4, "DUAS CAMADAS: XOR E O COLAPSO SEM NAO LINEARIDADE")

def camada(X, W, b, ativacao):
    """Camada densa: W tem formato (neuronios, entradas); b, (neuronios,)."""
    return ATIVACOES[ativacao](np.asarray(X, dtype=float) @ W.T + b)

E = np.array(entradas2, dtype=float)

# TODO 4a: monte uma rede de duas camadas com degrau que calcule o XOR:
#          oculta com 2 neuronios (um OR e um NAND), saida com 1 neuronio (AND
#          das duas saidas ocultas). Defina W1, b1, W2, b2, imprima a saida da
#          camada oculta e a saida final para as quatro entradas
W1 = np.array([[1.0, 1.0], [-1.0, -1.0]])
b1 = np.array([-0.5, 1.5])
W2 = np.array([[1.0, 1.0]])
b2 = np.array([-1.5])

H = camada(E, W1, b1, "degrau")
Y = camada(H, W2, b2, "degrau")
print("Camada oculta H (OR, NAND):\n", H.astype(int))
print("Saida final Y (XOR):", Y.ravel().astype(int).tolist())

# TODO 4b: repita o calculo trocando TODAS as ativacoes por "identidade".
#          Imprima as quatro saidas, a rede colapsada W' = W2 @ W1 e
#          b' = W2 @ b1 + b2, e confira com np.allclose que E @ W'.T + b'
#          reproduz as saidas das duas camadas
Hi = camada(E, W1, b1, "identidade")
Yi = camada(Hi, W2, b2, "identidade")
W_colapsado = W2 @ W1
b_colapsado = W2 @ b1 + b2
saida_colapsada = E @ W_colapsado.T + b_colapsado

print("Saidas com identidade:", Yi.ravel().tolist())
print("W' =", W_colapsado.tolist(), " b' =", b_colapsado.tolist())
print("np.allclose(Yi, colapsada):", np.allclose(Yi, saida_colapsada))

# TODO 4c: agora com ReLU na camada oculta e identidade na saida, resolva o XOR
#          com W1 = [[1, 1], [1, 1]], b1 = [0, -1] e escolha W2 (formato (1, 2))
#          e b2 para que a saida seja [0, 1, 1, 0]. Imprima a camada oculta e
#          a saida
W1_relu = np.array([[1.0, 1.0], [1.0, 1.0]])
b1_relu = np.array([0.0, -1.0])
W2_relu = np.array([[1.0, -2.0]])
b2_relu = np.array([0.0])

H_relu = camada(E, W1_relu, b1_relu, "relu")
Y_relu = camada(H_relu, W2_relu, b2_relu, "identidade")
print("Camada oculta ReLU:\n", H_relu)
print("Saida XOR (identidade):", Y_relu.ravel().tolist())

# ============================================================ EXERCICIO 5 (1,5)
secao(5, "SATURACAO, GRADIENTE E RELU MORTA")

# TODO 5a: escreva cadeia(x0, w, L, ativacao): L camadas de UM neuronio cada,
#          a_l = phi(w * a_(l-1)) (sem vies), e devolva (a_L, d a_L / d x0),
#          em que a derivada e o produto de w * phi'(z_l) em todas as camadas
#          (use DERIVADAS do exercicio 2)
def cadeia(x0, w, L, ativacao):
    a, grad = float(x0), 1.0
    for _ in range(L):
        z = w * a
        grad *= w * DERIVADAS[ativacao](z)
        a = float(ATIVACOES[ativacao](z))
    return a, grad

# TODO 5b: com x0 = 0,5, imprima |d a_L / d x0| para L em [1, 5, 10, 20, 50] e
#          para sigmoide, tanh, relu e leaky_relu, duas vezes: com w = 1,0 e
#          com w = 3,0 (use  df.to_string(float_format=lambda v: f"{v:.3g}"))
Ls_5b = [1, 5, 10, 20, 50]
ativacoes_5b = ["sigmoide", "tanh", "relu", "leaky_relu"]
for w_val in [1.0, 3.0]:
    linhas_5b = []
    for L_val in Ls_5b:
        linha = {"L": L_val}
        for nome in ativacoes_5b:
            _, grad = cadeia(0.5, w_val, L_val, nome)
            linha[nome] = abs(grad)
        linhas_5b.append(linha)
    df_5b = pd.DataFrame(linhas_5b).set_index("L")
    print(f"\nw = {w_val:.1f}:")
    print(df_5b.to_string(float_format=lambda v: f"{v:.3g}"))

# TODO 5c: gere Xu = rng.random((1000, 2)) com rng = np.random.default_rng(SEMENTE).
#          Para o neuronio w = (1; 1) e b em [0, -0,5, -1, -1,5, -2, -3], imprima
#          a fracao de entradas com z > 0 (neuronio ATIVO da ReLU), o gradiente
#          medio da ReLU e o gradiente medio da Leaky ReLU (3 casas)
rng = np.random.default_rng(SEMENTE)
Xu = rng.random((1000, 2))
w_5c = np.array([1.0, 1.0])
bs_5c = [0.0, -0.5, -1.0, -1.5, -2.0, -3.0]
linhas_5c = []
for b_val in bs_5c:
    z_u = Xu @ w_5c + b_val
    linhas_5c.append({
        "b": b_val,
        "frac_ativo": np.mean(z_u > 0),
        "grad_medio_relu": np.mean(d_relu(z_u)),
        "grad_medio_leaky": np.mean(d_leaky_relu(z_u)),
    })
df_5c = pd.DataFrame(linhas_5c).set_index("b")
print()
print(df_5c.round(3).to_string())

# ============================================================ EXERCICIO 6 (1,5)
secao(6, "O NEURONIO NA IRIS: REGRESSAO LOGISTICA E LINEAR")
iris = load_iris()
mascara = iris.target > 0                       # versicolor (0) x virginica (1)
Xi = StandardScaler().fit_transform(iris.data[mascara][:, [2, 3]])   # comprimento e largura da petala
yi = iris.target[mascara] - 1

# TODO 6a: ajuste LogisticRegression(random_state=SEMENTE) em (Xi, yi). Monte um
#          Neuronio com ativacao "sigmoide", w = lr.coef_[0] e b = lr.intercept_[0].
#          Imprima w, b e a MAIOR diferenca absoluta entre a saida do seu
#          neuronio e lr.predict_proba(Xi)[:, 1]
lr = LogisticRegression(random_state=SEMENTE).fit(Xi, yi)
w_lr = lr.coef_[0]
b_lr = lr.intercept_[0]
neuronio_lr = Neuronio(w_lr, b_lr, "sigmoide")
saidas_neuronio_lr = neuronio_lr.saida(Xi)
probs_sklearn = lr.predict_proba(Xi)[:, 1]
dif_max_lr = np.max(np.abs(saidas_neuronio_lr - probs_sklearn))
print(f"w: {w_lr.round(4)}, b: {b_lr:.4f}")
print(f"Maior diferenca absoluta (Neuronio vs sklearn): {dif_max_lr:.2e}")

# TODO 6b: com esses MESMOS w e b, classifique com tres neuronios: sigmoide
#          (classe 1 se y >= 0,5), degrau (y == 1) e tanh (classe 1 se y >= 0).
#          Imprima a acuracia de cada um e em quantas amostras discordam da
#          sigmoide
pred_sig = (neuronio_lr.saida(Xi) >= 0.5).astype(int)
pred_deg = (Neuronio(w_lr, b_lr, "degrau").saida(Xi) == 1.0).astype(int)
pred_tanh = (Neuronio(w_lr, b_lr, "tanh").saida(Xi) >= 0.0).astype(int)

acc_sig = np.mean(pred_sig == yi)
acc_deg = np.mean(pred_deg == yi)
acc_tanh = np.mean(pred_tanh == yi)

disc_deg = np.sum(pred_deg != pred_sig)
disc_tanh = np.sum(pred_tanh != pred_sig)

print(f"Acuracia Sigmoide: {acc_sig:.3f}")
print(f"Acuracia Degrau:   {acc_deg:.3f} | Discordancias com Sigmoide: {disc_deg}")
print(f"Acuracia Tanh:     {acc_tanh:.3f} | Discordancias com Sigmoide: {disc_tanh}")

# TODO 6c: ajuste LinearRegression() em (Xi, yi) e monte um neuronio de ativacao
#          "identidade" com esses coeficientes. Imprima a acuracia (classe 1 se
#          y >= 0,5), quantas saidas ficam FORA de [0, 1] (e o minimo e o
#          maximo) e quantas saidas da sigmoide ficam fora de [0, 1]
lin_reg = LinearRegression().fit(Xi, yi)
w_lin = lin_reg.coef_
b_lin = lin_reg.intercept_
neuronio_lin = Neuronio(w_lin, b_lin, "identidade")
saidas_lin = neuronio_lin.saida(Xi)

pred_lin = (saidas_lin >= 0.5).astype(int)
acc_lin = np.mean(pred_lin == yi)

fora_lin = np.sum((saidas_lin < 0) | (saidas_lin > 1))
min_lin = np.min(saidas_lin)
max_lin = np.max(saidas_lin)

fora_sig = np.sum((saidas_neuronio_lr < 0) | (saidas_neuronio_lr > 1))

print(f"Acuracia Regressao Linear (y >= 0.5): {acc_lin:.3f}")
print(f"Saidas da Regressao Linear fora de [0, 1]: {fora_lin} (min = {min_lin:.3f}, max = {max_lin:.3f})")
print(f"Saidas da Sigmoide fora de [0, 1]: {fora_sig}")

# ============================================================ EXERCICIO 7 (1,0)
secao(7, "SOFTMAX: ESTABILIDADE E TEMPERATURA")

# TODO 7a: escreva softmax_ingenua(z) = exp(z) / soma e softmax(z, T=1.0), que
#          divide z por T e subtrai o maximo antes do exp. Teste as duas em
#          z = (1000, 1001, 1002) (envolva a ingenua em np.errstate(all="ignore"))
#          e a estavel tambem em (1, 2, 3)
def softmax_ingenua(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z)
    return e / np.sum(e)

def softmax(z, T=1.0):
    z = np.asarray(z, dtype=float) / T
    e = np.exp(z - np.max(z))
    return e / np.sum(e)

z_grande = np.array([1000.0, 1001.0, 1002.0])
with np.errstate(all="ignore"):
    s_ingenua = softmax_ingenua(z_grande)
s_estavel = softmax(z_grande, T=1.0)
s_123 = softmax(np.array([1.0, 2.0, 3.0]), T=1.0)

print("z = [1000, 1001, 1002]:")
print("  softmax ingenua:", s_ingenua)
print("  softmax estavel:", np.round(s_estavel, 4).tolist())
print("z = [1, 2, 3]:")
print("  softmax estavel:", np.round(s_123, 4).tolist())

# TODO 7b: com z0 = (3; 1; 0) e T em [0,5, 1, 2, 10], imprima as probabilidades,
#          a soma e a entropia -sum(p * ln(p)). Imprima tambem ln(3)
z0 = np.array([3.0, 1.0, 0.0])
for T_val in [0.5, 1.0, 2.0, 10.0]:
    p = softmax(z0, T=T_val)
    soma = np.sum(p)
    entropia = -np.sum(p * np.log(p))
    print(f"T = {T_val:>4.1f} | p = {np.round(p, 4).tolist()} | soma = {soma:.4f} | entropia = {entropia:.4f}")
print(f"ln(3) = {np.log(3):.4f}")

# TODO 7c: confira numericamente que softmax((2; 0,5))[0] e igual a
#          sigmoide(2 - 0,5)
p_sm = softmax(np.array([2.0, 0.5]))[0]
p_sig = sigmoide(2.0 - 0.5)
print(f"softmax([2, 0.5])[0] = {p_sm:.6f}")
print(f"sigmoide(2 - 0.5)    = {p_sig:.6f}")
print(f"Iguais: {np.isclose(p_sm, p_sig)}")

# ============================================================ RELATORIO
"""
=======================================================================
RELATORIO  (vale 60% da nota desta sessao)
=======================================================================

Ex.1 - (a) Qual e a unica diferenca de codigo entre calcular a saida de
       UM vetor e de um LOTE? (b) Por que o mesmo z produz saidas tao
       diferentes conforme a ativacao? Use os numeros da sua tabela.
    R: (a) Nao ha diferenca no codigo: o @ (ou np.dot) realiza tanto o produto
       escalar para um unico vetor (d,) quanto o produto matriz-vetor para um lote (n, d),
       com a adicao do b sendo transmitida por broadcasting em ambos os casos.
       (b) Porque cada ativacao aplica uma transformacao matematica diferente ao valor z.
       Para z = -1.0, o degrau resulta em 0.0 (pois z < 0); a identidade mantem -1.0; a
       sigmoide comprime suavemente em 0.269; a tanh comprime simetricamente em -0.762;
       a ReLU anula valores negativos resultando em 0.0; e a Leaky ReLU aplica uma
       inclinacao residual de alpha = 0.01, resultando em -0.010.

Ex.2 - (a) Quais ativacoes tem media proxima de zero em [-10, 10] e por
       que isso importa? (b) Qual fracao de pontos fica saturada em cada
       ativacao e o que isso significa para o treinamento? (c) Por que o
       degrau nao serve para treinar por gradiente?
    R: (a) Apenas a tangente hiperbolica (tanh) tem media proxima de zero (0.000), pois e
       uma funcao impar simetrica em relacao a origem. Essa propriedade e importante para
       manter os gradientes centrados, evitando que as atualizacoes dos pesos sofram com
       vies sistematico em uma unica direcao (efeito zig-zag) durante a otimizacao.
       (b) Fracoes saturadas (derivada < 0.01): Degrau = 1.000 (100%); Tanh = 0.701 (70.1%);
       Sigmoide = 0.542 (54.2%); ReLU = 0.500 (50.0%); Leaky ReLU = 0.000 (0.0%).
       Para o treinamento, pontos saturados possuem derivadas proximas de zero, o que anula
       ou enfraquece o gradiente na retropropagacao (gradiente que desaparece), travando o
       aprendizado dos pesos das camadas anteriores.
       (c) A funcao degrau tem derivada nula em quase todo o dominio e e descontinua em z = 0.
       Como o gradiente do erro em relacao aos pesos e proporcional a derivada da ativacao,
       a derivada zero faz com que o gradiente seja sempre nulo, impedindo que algoritmos
       baseados em gradiente ajustem os pesos.

Ex.3 - (a) Por que AND, OR, NOT e MAJ3 tem solucao e XOR/XNOR nao tem
       nenhuma na grade? (b) O que a multiplicacao dos pesos por k = 10
       fez com a saida sigmoide? Em que situacao ela se aproxima do degrau?
    R: (a) Porque AND, OR, NOT e MAJ3 sao problemas linearmente separaveis, nos quais um unico
       hiperplano (ou reta) w @ x + b = 0 consegue isolar as saidas 1 das saidas 0.
       XOR e XNOR nao sao linearmente separaveis, pois as classes ocupam posicoes diagonais
       opostas que nao podem ser separadas por uma unica reta.
       (b) Multiplicar por k = 10 amplificou os valores de z, empurrando as respostas para as
       regioes de saturacao da sigmoide e aproximando as saidas de 0 ou 1 (0.0000, 0.0067 e 0.9933).
       A sigmoide se aproxima do degrau quando a escala dos pesos tende ao infinito (k -> inf),
       tornando a curva de transicao infinitamente inclinada em z = 0.

Ex.4 - (a) Por que a rede de 4b devolve a mesma saida para as quatro
       entradas? (b) O que a camada oculta do 4a "faz" com os pontos para
       que o neuronio de saida consiga separa-los? (c) Por que a ReLU na
       oculta resolve o XOR e a identidade nao?
    R: (a) Porque a composicao de transformacoes lineares (identidade) e puramente linear.
       A rede colapsa em W' = W2 @ W1 = [[0, 0]] e b' = W2 @ b1 + b2 = [-0.5], anulando o efeito
       das entradas e gerando -0.5 para qualquer entrada.
       (b) A camada oculta projeta as entradas no espaco intermediario (OR, NAND). Nesse espaco,
       os pontos (0,1) e (1,0) sao mapeados para (1,1), enquanto (0,0) vai para (0,1) e (1,1) vai
       para (1,0). Com essa reorganizacao geometrica, as classes tornam-se linearmente separaveis
       por um neuronio AND na saida.
       (c) A identidade preserva a linearidade do espaco, mantendo o problema nao separavel.
       A ReLU introduz uma nao linearidade (corte em z = 0) que dobra o espaco de atributos,
       permitindo que a combinacao linear da camada seguinte separe as classes do XOR.

Ex.5 - (a) Descreva o que acontece com o gradiente da sigmoide e da tanh
       quando L cresce, com w = 1 e com w = 3. (b) A ReLU resolve o
       problema? O que acontece com ela quando w = 3? (c) O que a tabela
       do item 5c mostra sobre a ReLU morta e como a Leaky ReLU ajuda?
    R: (a) Com w = 1.0, o gradiente da sigmoide e da tanh decai exponencialmente para zero a
       medida que L aumenta (gradiente que desaparece). Com w = 3.0, a saturacao das derivadas
       nas pontas faz o gradiente colapsar ainda mais rapidamente (chegando a 3.01e-40 e 5.20e-75 em L=50).
       (b) A ReLU resolve o desaparecimento para neuronios ativos porque sua derivada vale 1.
       Contudo, quando w = 3.0, o gradiente e multiplicado por 3 em cada camada, explodindo
       exponencialmente para 7.18e+23 em L = 50 (problema do gradiente que explode).
       (c) A tabela mostra que quando o vies b fica muito negativo (b <= -2.0), 0% das entradas
       ficam ativas (frac_ativo = 0.0), zerando o gradiente da ReLU (neuronio morto). A Leaky ReLU
       mantem um gradiente residual constante de 0.010, permitindo que o neuronio continue
       recebendo atualizacoes e possa voltar a se ativar.

Ex.6 - (a) O que a diferenca ~0 do item 6a mostra sobre regressao logistica
       e neuronio? (b) Se a fronteira e a mesma nas tres ativacoes, para
       que serve a sigmoide? (c) Qual problema aparece nas saidas do
       neuronio identidade quando a resposta e uma probabilidade?
    R: (a) Demonstra que a regressao logistica binaria e identica a um unico neuronio artificial
       com funcao de ativacao sigmoide, coeficientes w e intercepto b.
       (b) Embora a fronteira rigida de decisao z = 0 seja a mesma, a sigmoide produz probabilidades
       calibradas no intervalo continuo (0, 1). Isso viabiliza expressar o grau de certeza da previsao,
       ajustar limiares de decisao com base em custos de erro e treinar por entropia cruzada.
       (c) A identidade produz valores irrestritos em toda a reta real. No experimento, 28 saidas
       ficaram fora do intervalo valido [0, 1] (variando de -0.266 a 1.308), violando a interpretacao
       de probabilidade.

Ex.7 - (a) Por que a softmax ingenua falha e a estavel nao? (b) Como a
       temperatura muda a distribuicao e a entropia? (c) O que o item 7c
       diz sobre a sigmoide e a softmax?
    R: (a) A softmax ingenua calcula exp(z), gerando overflow numerico (inf) para pontuacoes grandes
       como 1000, o que resulta em inf / inf = nan. A versao estavel subtrai o maximo antes do exp,
       garantindo expoentes <= 0 e evitando overflow sem alterar as probabilidades.
       (b) Temperaturas maiores (T > 1) suavizam as probabilidades, aproximando-as de uma distribuicao
       uniforme e elevando a entropia ate o limite ln(3) = 1.0986. Temperaturas menores (T < 1) tornam
       a distribuicao mais concentrada no valor maximo, diminuindo a entropia.
       (c) Mostra que a sigmoide e equivalente a softmax para o caso de duas classes: a probabilidade
       de uma classe via softmax de duas pontuacoes e exatamente a sigmoide aplicada a diferenca
       entre essas pontuacoes, sigma(z1 - z2).
=======================================================================
"""

print("""
=======================================================================
Arquivo executado. Confira se todos os TODO foram preenchidos e se o
bloco RELATORIO esta respondido antes de entregar.
=======================================================================""")
