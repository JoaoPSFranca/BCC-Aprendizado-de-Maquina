"""
Aprendizado de Maquina - IFSP Presidente Epitacio
Aula 7, Parte 2 - MINI-LABORATORIO  ***ARQUIVO DO ALUNO***

NOME: ______________________________  PRONTUARIO: ______________  DATA: __/__/2026

INSTRUCOES
  - Preencha os trechos marcados com  # TODO
  - Use SEMPRE random_state=42
  - Rode o arquivo ate ele executar sem erro:   python lab07_parte2_exercicios.py
  - Responda as perguntas de interpretacao no bloco RELATORIO, no fim do arquivo
  - Entregue este arquivo executado (e a figura gerada no Exercicio 3)

VALE 10,0 PONTOS - integra a nota do Trabalho 2 (Pratico)
  codigo 30%  |  relatorio 70%  (sessao mais conceitual que as anteriores)
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

SEMENTE = 42


def secao(n, titulo):
    print()
    print("=" * 70)
    print(f"EXERCICIO {n} - {titulo}")
    print("=" * 70)


# ============================================================ EXERCICIO 1 (2,0)
secao(1, "RECONHECENDO AS 5 FAMILIAS PELA IDEIA CENTRAL")

# TODO 1: sem codigo. Cada descricao abaixo corresponde a UMA das 5
#         familias vistas hoje (k-NN, Regressao Logistica, Arvore de
#         Decisao, Naive Bayes, SVM). Identifique qual e qual no RELATORIO:
#   (a) "Calcula uma probabilidade com uma formula linear passada por uma
#        curva em S; a fronteira final e (aproximadamente) uma reta."
#   (b) "Faz uma serie de perguntas do tipo 'atributo X e maior que
#        valor V?', e a fronteira final parece uma escadinha de
#        retangulos."
#   (c) "Olha para os vizinhos mais proximos do ponto e vota a maioria;
#        nao existe 'treinamento' de coeficientes, so memorizacao dos
#        dados."
#   (d) "Usa o teorema de Bayes supondo (de forma simplificadora) que os
#        atributos sao independentes entre si, dado a classe."
#   (e) "Busca a fronteira que maximiza a margem entre as classes; com
#        kernel nao linear, a fronteira pode virar uma curva complexa."


# ============================================================ EXERCICIO 2 (3,0)
secao(2, "OS MESMOS 5 MODELOS, COM MAIS RUIDO")

# O dataset de hoje (demonstracao) usou noise=0.25. Agora vamos repetir
# com noise=0.40 - as duas "luas" ficam bem mais misturadas.

# TODO 2a: gere X, y com make_moons(n_samples=300, noise=0.40, random_state=SEMENTE)

# TODO 2b: divida em treino/teste com train_test_split(test_size=0.30, random_state=SEMENTE)

# TODO 2c: treine os MESMOS 5 modelos da demonstracao (pode copiar o
#          dicionario "modelos" do lab07_parte2_demo.py) nesse novo treino

# TODO 2d: imprima a acuracia de cada um no novo teste, no mesmo formato
#          da demonstracao


# ============================================================ EXERCICIO 3 (2,0)
secao(3, "A FIGURA DE FRONTEIRAS, COM MAIS RUIDO")

# TODO 3: adapte o codigo do BLOCO 5 do lab07_parte2_demo.py (a figura
#         com os 5 paineis) para o dataset do Exercicio 2, e salve como
#         fronteiras_decisao_ruido040.png


# ============================================================ EXERCICIO 4 (1,5)
secao(4, "UM PONTO DIFICIL DE VERDADE")

# TODO 4a: escolha um ponto (x1, x2) que fique bem PROXIMO da fronteira
#          entre as duas "luas" no grafico do Exercicio 3 (so olhar a
#          figura e escolher visualmente, sem calculo)

# TODO 4b: imprima como cada um dos 5 modelos classifica esse ponto,
#          igual ao BLOCO 4 do lab07_parte2_demo.py


# ============================================================ EXERCICIO 5 (1,5)
secao(5, "SVM E SVR: A MESMA IDEIA, DOIS PROBLEMAS DIFERENTES")

# TODO 5: sem codigo. Responda no RELATORIO, relembrando o SVR da Aula 6.


p