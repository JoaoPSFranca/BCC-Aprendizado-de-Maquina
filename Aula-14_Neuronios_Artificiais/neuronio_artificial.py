"""
Simulação de um neurônio artificial.

Um neurônio calcula, em duas etapas:
    z = w1*x1 + w2*x2 + ... + wn*xn + b      (soma ponderada + viés)
    y = phi(z)                               (função de ativação)

Requer apenas NumPy. O gráfico final usa Matplotlib (opcional).
Execute com:  python neuronio_artificial.py
"""

import numpy as np


# ---------------------------------------------------------------------------
# Funções de ativação e suas derivadas
# ---------------------------------------------------------------------------
def degrau(z):
    return np.where(z >= 0, 1.0, 0.0)


def identidade(z):
    return z


def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))


def tanh(z):
    return np.tanh(z)


def relu(z):
    return np.maximum(0.0, z)


def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)


# Derivadas (usadas no treinamento por gradiente; o degrau tem derivada nula)
def d_sigmoide(z):
    s = sigmoide(z)
    return s * (1.0 - s)


def d_tanh(z):
    return 1.0 - np.tanh(z) ** 2


def d_relu(z):
    return np.where(z > 0, 1.0, 0.0)


def softmax(z):
    """Transforma um vetor de pontuações em probabilidades que somam 1."""
    e = np.exp(z - np.max(z))  # subtrair o máximo evita overflow
    return e / e.sum()


ATIVACOES = {
    "degrau": degrau,
    "identidade": identidade,
    "sigmoide": sigmoide,
    "tanh": tanh,
    "relu": relu,
    "leaky_relu": leaky_relu,
}


# ---------------------------------------------------------------------------
# O neurônio
# ---------------------------------------------------------------------------
class Neuronio:
    def __init__(self, pesos, vies=0.0, ativacao="sigmoide"):
        self.w = np.asarray(pesos, dtype=float)
        self.b = float(vies)
        self.nome_ativacao = ativacao
        self.phi = ATIVACOES[ativacao]

    def soma_ponderada(self, x):
        """z = w . x + b"""
        return float(np.dot(self.w, np.asarray(x, dtype=float)) + self.b)

    def saida(self, x):
        """y = phi(z)"""
        return float(self.phi(self.soma_ponderada(x)))

    def __repr__(self):
        return f"Neuronio(w={self.w.tolist()}, b={self.b}, ativacao='{self.nome_ativacao}')"


# ---------------------------------------------------------------------------
# Demonstrações
# ---------------------------------------------------------------------------
def demo_exemplo_a_mao():
    print("=" * 70)
    print("1) Exemplo feito à mão: mesma soma z, saídas diferentes por ativação")
    print("=" * 70)
    x = [2, -1, 3]
    w = [0.5, 1.0, -0.2]
    b = 0.3
    for nome in ATIVACOES:
        n = Neuronio(w, b, nome)
        print(f"  {nome:<11} z = {n.soma_ponderada(x):+.3f}   y = {n.saida(x):+.3f}")


def demo_portas_logicas():
    print()
    print("=" * 70)
    print("2) Portas lógicas com um neurônio de degrau")
    print("=" * 70)
    entradas = [(0, 0), (0, 1), (1, 0), (1, 1)]
    portas = {
        "AND": ([1, 1], -1.5),
        "OR": ([1, 1], -0.5),
        "NAND": ([-1, -1], 1.5),
        "NOR": ([-1, -1], 0.5),
    }
    for nome, (w, b) in portas.items():
        n = Neuronio(w, b, "degrau")
        saidas = [int(n.saida(x)) for x in entradas]
        print(f"  {nome:<5} w={w} b={b:+}   saídas para (00,01,10,11): {saidas}")

    # O XOR não é separável por uma reta: nenhum (w1, w2, b) acerta as quatro linhas.
    alvo = [0, 1, 1, 0]
    achou = False
    grade = np.arange(-3, 3.01, 0.25)
    for w1 in grade:
        for w2 in grade:
            for b in grade:
                n = Neuronio([w1, w2], b, "degrau")
                if [int(n.saida(x)) for x in entradas] == alvo:
                    achou = True
    print(f"  XOR   busca em grade de pesos e viés encontrou solução? {achou}")


def demo_saturacao():
    print()
    print("=" * 70)
    print("3) Saturação: derivadas em função de z")
    print("=" * 70)
    zs = np.array([-4, -2, -1, 0, 1, 2, 4], dtype=float)
    print(f"  {'z':>4} {'sigmoide':>9} {'derivada':>9} {'tanh':>8} {'derivada':>9} {'relu':>6} {'derivada':>9}")
    for z in zs:
        print(f"  {z:>4.0f} {sigmoide(z):>9.3f} {d_sigmoide(z):>9.3f} {tanh(z):>8.3f} "
              f"{d_tanh(z):>9.3f} {relu(z):>6.1f} {d_relu(z):>9.1f}")
    print(f"\n  Produto máximo de 10 derivadas da sigmoide: 0,25^10 = {0.25 ** 10:.2e}")


def demo_softmax():
    print()
    print("=" * 70)
    print("4) Softmax")
    print("=" * 70)
    z = np.array([2.0, 1.0, 0.1])
    p = softmax(z)
    print(f"  pontuações {z.tolist()} -> probabilidades {np.round(p, 3).tolist()} (soma = {p.sum():.1f})")
    print(f"  somando 10 a todas: {np.round(softmax(z + 10), 3).tolist()} (não muda)")
    print(f"  multiplicando por 2: {np.round(softmax(z * 2), 3).tolist()} (mais concentrada)")


def grafico_ativacoes():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nMatplotlib não instalado: gráfico ignorado.")
        return
    z = np.linspace(-6, 6, 400)
    fig, eixos = plt.subplots(1, 5, figsize=(14, 2.8), sharex=True)
    for ax, (nome, f) in zip(eixos, [("degrau", degrau), ("sigmoide", sigmoide), ("tanh", tanh),
                                      ("relu", relu), ("leaky_relu", leaky_relu)]):
        ax.plot(z, f(z), lw=2)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_title(nome)
        ax.set_xlabel("z")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_exemplo_a_mao()
    demo_portas_logicas()
    demo_saturacao()
    demo_softmax()
    grafico_ativacoes()
