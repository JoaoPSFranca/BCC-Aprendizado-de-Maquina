"""
Aprendizado de Máquina - IFSP
Conversão de Temperatura (Celsius para Fahrenheit) utilizando
Rede Neural / Perceptron de 1 Neurônio (Ativação Linear / Regressão)

Fórmula teórica: F = C * 1.8 + 32
A rede deve aprender:
  - Peso (w) ≈ 1.8
  - Bias (b) ≈ 32.0
"""

import sys

try:
    import numpy as np
    TEM_NUMPY = True
except ImportError:
    TEM_NUMPY = False


class PerceptronRegressao:
    """
    Perceptron de 1 neurônio com função de ativação linear (identidade),
    também conhecido como Adaline para regressão linear.
    
    Saída:
        y_hat = w * x + b
    """

    def __init__(self, taxa_aprendizado=0.001, epocas=8000):
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.w = 0.0
        self.b = 0.0
        self.historico_perda = []

    def prever(self, x):
        """Calcula a saída do neurônio: y = w * x + b"""
        if isinstance(x, (list, tuple)):
            return [self.w * item + self.b for item in x]
        if TEM_NUMPY and isinstance(x, np.ndarray):
            return self.w * x + self.b
        return self.w * float(x) + self.b

    def treinar(self, X, y, verbose=True):
        """
        Treina o neurônio usando Descida do Gradiente em Lote (Batch Gradient Descent).
        
        Função de custo (MSE):
            L = (1/N) * sum((y_i - (w * x_i + b))^2)
            
        Gradientes:
            dL/dw = -2/N * sum((y_i - y_hat_i) * x_i)
            dL/db = -2/N * sum(y_i - y_hat_i)
            
        Regra de atualização:
            w = w - eta * dL/dw
            b = b - eta * dL/db
        """
        X_lista = list(X)
        y_lista = list(y)
        n = len(X_lista)

        self.w = 0.0
        self.b = 0.0
        self.historico_perda = []

        if verbose:
            print("Iniciando treinamento do Perceptron de 1 Neurônio...")
            print(f"Taxa de aprendizado (eta): {self.taxa_aprendizado} | Épocas: {self.epocas}\n")

        for epoca in range(1, self.epocas + 1):
            erros = [y_i - (self.w * x_i + self.b) for x_i, y_i in zip(X_lista, y_lista)]
            mse = sum(e ** 2 for e in erros) / n
            self.historico_perda.append(mse)

            grad_w = -2.0 * sum(e * x_i for e, x_i in zip(erros, X_lista)) / n
            grad_b = -2.0 * sum(erros) / n

            self.w -= self.taxa_aprendizado * grad_w
            self.b -= self.taxa_aprendizado * grad_b

            if verbose and (epoca % 1000 == 0 or epoca == 1 or epoca == self.epocas):
                print(f"Época {epoca:5d}/{self.epocas} -> MSE: {mse:10.5f} | w: {self.w:8.4f} | b: {self.b:8.4f}")

        if verbose:
            print("\nTreinamento concluído com sucesso!")
            print(f"Valores aprendidos pelo neurônio: w = {self.w:.4f} (esperado: 1.8), b = {self.b:.4f} (esperado: 32.0)")

        return self


def main():
    print("=" * 70)
    print("CONVERSOR CELSIUS -> FAHRENHEIT COM PERCEPTRON DE 1 NEURÔNIO")
    print("=" * 70)

    # Conjunto de dados de treino: temperaturas em Celsius e em Fahrenheit correspondentes
    celsius_treino = [-40.0, -10.0, 0.0, 8.0, 15.0, 22.0, 38.0]
    fahrenheit_treino = [-40.0, 14.0, 32.0, 46.4, 59.0, 71.6, 100.4]

    print("\nDados de Treinamento:")
    for c, f in zip(celsius_treino, fahrenheit_treino):
        print(f"  {c:6.1f} °C  ->  {f:6.1f} °F")
    print()

    # Instanciando e treinando o perceptron
    modelo = PerceptronRegressao(taxa_aprendizado=0.001, epocas=8000)
    modelo.treinar(celsius_treino, fahrenheit_treino, verbose=True)

    print("\n" + "=" * 70)
    print("TESTES COM NOVAS TEMPERATURAS (DADOS NÃO VISTOS NO TREINO)")
    print("=" * 70)

    celsius_teste = [-20.0, 10.0, 25.0, 30.0, 100.0]
    for c in celsius_teste:
        f_pred = modelo.prever(c)
        f_real = c * 1.8 + 32.0
        erro = abs(f_pred - f_real)
        print(f"Celsius: {c:6.1f} °C | Previsto pelo Perceptron: {f_pred:6.2f} °F | Real (Fórmula): {f_real:6.2f} °F | Erro: {erro:6.4f}")

    print("\n" + "=" * 70)
    print("DEMONSTRAÇÃO DA FÓRMULA APRENDIDA PELO NEURÔNIO:")
    print(f"  Fahrenheit = {modelo.w:.4f} * Celsius + {modelo.b:.4f}")
    print("=" * 70)

    # Se o usuário passou um valor por argumento na linha de comando, calcula a conversão direta
    if len(sys.argv) > 1:
        try:
            c_input = float(sys.argv[1])
            f_calc = modelo.prever(c_input)
            print(f"\nConversão informada por argumento: {c_input:.2f} °C -> {f_calc:.2f} °F")
        except ValueError:
            pass


if __name__ == "__main__":
    main()
