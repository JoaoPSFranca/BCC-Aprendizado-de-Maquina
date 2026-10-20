# BCC - Aprendizado de Máquina

Repositório destinado ao armazenamento e organização dos códigos, laboratórios, apostilas e avaliações da disciplina de **Aprendizado de Máquina** do curso de **Bacharelado em Ciência da Computação (BCC)** — Instituto Federal de São Paulo (IFSP), Câmpus Presidente Epitácio.

---

## 📁 Estrutura do Repositório

O projeto está organizado em módulos temáticos sequenciais (`Aula-XX_materia`), além de pastas dedicadas para atividades práticas e avaliações:

```text
BCC-Aprendizado-de-Maquina/
├── Aula-01_Introducao_e_KNN/
├── Aula-02_KNN_e_Escalonamento/
├── Aula-03_Numpy_Pandas_Extracao_X_y/
├── Aula-04_KMeans_Nao_Supervisionado/
├── Aula-05_KMeans_Padronizacao/
├── Aula-06_Preparacao_e_Refinamento_de_Dados/
├── Aula-07_Similaridade_PCA_e_Classificacao/
├── Aula-08_Modelos_de_Regressao/
├── Aula-09_Metricas_Regressao_e_Residuos/
├── Aula-10_Algoritmos_de_Classificacao/
├── Aula-11_Comparacao_de_Classificadores/
├── Aula-12_Validacao_Honesta_e_Overfitting/
├── Aula-13_Comparacao_Geral_de_Modelos/
├── atividades/
│   ├── atividade_01_limpeza_dados/
│   ├── atividade_dupla_knn_kmeans/
│   └── atividade_tres_bases/
├── avaliacoes/
│   ├── prova_01/
│   └── prova_02/
├── revisao_aulas_01_a_12.pdf
└── README.md
```

---

## 📚 Conteúdo por Aula

| Pasta | Descrição e Tópicos Abordados |
| :--- | :--- |
| **Aula-01_Introducao_e_KNN** | Introdução ao Aprendizado de Máquina, inspeção inicial de bases e primeiro classificador K-NN (datasets de clientes e frutas). |
| **Aula-02_KNN_e_Escalonamento** | Notações formais, efeito do hiperparâmetro $k$, técnicas de escalonamento de atributos e laboratório com Iris. |
| **Aula-03_Numpy_Pandas_Extracao_X_y** | Do problema real a $X$ e $y$: manipulação vetorial com NumPy e Pandas, matriz de atributos e vetor de rótulos. |
| **Aula-04_KMeans_Nao_Supervisionado** | Paradigma não-supervisionado: algoritmo K-Means, centróides e método do cotovelo (*Elbow Method*). |
| **Aula-05_KMeans_Padronizacao** | Impacto da escala e padronização das variáveis na formação de agrupamentos no K-Means. |
| **Aula-06_Preparacao_e_Refinamento_de_Dados** | Caracterização, limpeza, geração de dados sintéticos, pipelines do scikit-learn e divisão treino/teste. |
| **Aula-07_Similaridade_PCA_e_Classificacao** | Matrizes de similaridade/distância, geometria dos dados, redução de dimensionalidade (PCA, LDA, Kernel PCA) e classificação aplicada. |
| **Aula-08_Modelos_de_Regressao** | Modelos preditivos de regressão: Linear Simples, Múltipla, Polinomial e Support Vector Regression (SVR). |
| **Aula-09_Metricas_Regressao_e_Residuos** | Laboratório de regressão linear, análise de resíduos e métricas de desempenho (MAE, RMSE, $R^2$), com trabalho integrador de aluguéis. |
| **Aula-10_Algoritmos_de_Classificacao** | Modelos supervisionados de classificação: Naive Bayes, SVM, Regressão Logística, K-NN, Árvores de Decisão, Random Forest e análise de fronteiras. |
| **Aula-11_Comparacao_de_Classificadores** | Laboratório comparativo de 5 classificadores no mesmo problema real com Pipeline, ColumnTransformer e matrizes de confusão. |
| **Aula-12_Validacao_Honesta_e_Overfitting** | Holdout vs Validação Cruzada ($K$-Fold, Estratificado, Leave-One-Out), underfitting vs overfitting, curvas de aprendizado e prevenção de vazamento de dados (*data leakage*). |
| **Aula-13_Comparacao_Geral_de_Modelos** | Laboratório final unificado: protocolo padronizado de validação cruzada para comparação de todos os modelos supervisionados (regressão e classificação). |

---

## 🛠️ Atividades Práticas (`atividades/`)

- **`atividade_01_limpeza_dados/`**: Tratamento de dados brutos de vendas, resolução de inconsistências e resumo de transformações.
- **`atividade_dupla_knn_kmeans/`**: Atividade colaborativa prática em dupla envolvendo criação/tratamento de dados com regras de negócio (Dragon Ball Xenoverse 2 - DBXV2 e base de plantas), com aplicação de K-NN e K-Means.
- **`atividade_tres_bases/`**: Desafio prático integrador com três problemas distintos:
  - `resolucao_base1_classificacao_churn.ipynb`: Classificação Binária supervisionada de Churn (comparação de modelos, classes balanceadas, métricas F1/ROC-AUC e análise de custos).
  - `resolucao_base2_regressao_imoveis.ipynb`: Regressão supervisionada para previsão de preços de imóveis (tratamento de erro de digitação, métricas MAE/RMSE/R² e análise de resíduos).
  - `resolucao_base3_agrupamento_clientes.ipynb`: Aprendizado Não Supervisionado de Agrupamento/Clustering (seleção de features, varredura de K com Silhueta/Davies-Bouldin, K-Means vs Hierárquico, personas e projeção PCA 2D).

---

## 📝 Avaliações (`avaliacoes/`)

- **`prova_01/`**: Avaliação prática sobre aprendizado supervisionado e não supervisionado (datasets de linhas de ônibus e talhões de soja).
- **`prova_02/`**: Avaliação prática sobre regressão polinomial, K-NN, preparação de dados e prevenção de vazamento (datasets de estufas de energia e lotes de café).

---

## 📖 Revisão Geral

- **`revisao_aulas_01_a_12.pdf`**: Guia abrangente de arquitetura de dados, modelos e validação honesta cobrindo os conceitos das Aulas 1 a 12.
