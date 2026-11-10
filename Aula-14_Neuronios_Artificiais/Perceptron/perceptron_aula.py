"""
Aprendizado de Maquina - IFSP Presidente Epitacio
Aula - Perceptron de camada unica: o algoritmo, passo a passo
Script de apoio da apostila (o professor projeta e roda durante a aula).

Rode:  python perceptron_aula.py      (leva menos de um segundo; so precisa de NumPy)
"""
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
PORTAS = {
    "AND": np.array([0, 0, 0, 1]),
    "OR": np.array([0, 1, 1, 1]),
    "NOR": np.array([1, 0, 0, 0]),
    "XOR": np.array([0, 1, 1, 0]),
}


def degrau(z):
    return 1 if z >= 0 else 0


def treina_perceptron(X, t, eta=1.0, max_epocas=100, w0=None, b0=0.0, verbose=False):
    """Regra do perceptron. Devolve w, b e a lista de erros por epoca."""
    w = np.zeros(X.shape[1]) if w0 is None else np.array(w0, dtype=float)
    b = float(b0)
    erros_por_epoca = []
    for epoca in range(1, max_epocas + 1):
        erros = 0
        for x, alvo in zip(X, t):
            z = w @ x + b
            y_hat = degrau(z)
            e = alvo - y_hat
            if verbose:
                print(f"  epoca {epoca}  x={x.astype(int)}  t={alvo}  z={z:+.1f}  y={y_hat}  e={e:+d}", end="")
            if e != 0:
                w = w + eta * e * x
                b = b + eta * e
                erros += 1
                if verbose:
                    print(f"   ->  w={w}  b={b:+.1f}")
            elif verbose:
                print("   (acertou)")
        erros_por_epoca.append(erros)
        if erros == 0:
            break
    return w, b, erros_por_epoca


def secao(titulo):
    print()
    print("=" * 70)
    print(titulo)
    print("=" * 70)


# ---------------------------------------------------------------- 1. AND passo a passo
secao("1. AND, eta = 1, w = 0, b = 0: as duas primeiras epocas, exemplo por exemplo")
w, b, hist = treina_perceptron(X, PORTAS["AND"], eta=1.0, max_epocas=2, verbose=True)
print("\nTreino completo:")
w, b, hist = treina_perceptron(X, PORTAS["AND"], eta=1.0)
print(f"  erros por epoca: {hist}   ->   w = {w},  b = {b:+.0f}")
print("  conferencia (z, saida):", [(float(w @ x + b), degrau(w @ x + b)) for x in X])

# ---------------------------------------------------------------- 2. varias portas
secao("2. Portas separaveis convergem; o XOR nao")
for nome, t in PORTAS.items():
    w, b, hist = treina_perceptron(X, t, eta=1.0, max_epocas=15)
    print(f"  {nome:>3}: erros por epoca = {hist}")
    if hist[-1] == 0:
        print(f"       convergiu em {len(hist)} epocas:  w = {w},  b = {b:+.0f}")
    else:
        print("       NAO convergiu em 15 epocas (os pesos ficam em ciclo)")

# ---------------------------------------------------------------- 3. taxa de aprendizado
secao("3. Taxa de aprendizado com pesos iniciais nulos: so muda a escala")
for eta in (1.0, 0.5, 0.25):
    w, b, hist = treina_perceptron(X, PORTAS["OR"], eta=eta)
    print(f"  eta = {eta:<4}  erros por epoca = {hist}   w = {w}  b = {b:+.3f}")
print("  (mesma sequencia de erros; pesos finais proporcionais a eta)")

print("\nCuidado com empates z = 0 em ponto flutuante: eta = 0,1 nao e exato em binario")
w, b, hist = treina_perceptron(X, PORTAS["AND"], eta=0.1)
print(f"  AND com eta = 0.1: erros por epoca = {hist}   w = {w},  b = {b:+.3f}")
print("  (compare com a sequencia de eta = 1: 2, 3, 3, 2, 1, 0)")

# ---------------------------------------------------------------- 4. pesos iniciais
secao("4. Pesos iniciais nao nulos mudam o caminho (OR, eta = 1)")
w, b, hist = treina_perceptron(X, PORTAS["OR"], eta=1.0, w0=(0.3, -0.2), b0=0.1)
print(f"  erros por epoca = {hist}   w = {w},  b = {b:+.1f}")
