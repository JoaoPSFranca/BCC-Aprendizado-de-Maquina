import pandas as pd
import matplotlib.pyplot as plt

# Ler os arquivos CSV
df_linear = pd.read_csv('aluguel_praia_1.csv')
df_polinomial = pd.read_csv('aluguel_praia_2.csv')

# Criar figura com dois subplots lado a lado
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico 1: Dados Lineares
ax1.scatter(df_linear['distancia_praia_km'], df_linear['valor_aluguel'], alpha=0.5, color='blue')
ax1.set_xlabel('Distância da Praia (km)')
ax1.set_ylabel('Valor do Aluguel')
ax1.set_title('Aluguel - Modelo Linear')
ax1.grid(True, alpha=0.3)

# Gráfico 2: Dados Polinomiais
ax2.scatter(df_polinomial['distancia_praia_km'], df_polinomial['valor_aluguel'], alpha=0.5, color='red')
ax2.set_xlabel('Distância da Praia (km)')
ax2.set_ylabel('Valor do Aluguel')
ax2.set_title('Aluguel - Modelo Polinomial')
ax2.grid(True, alpha=0.3)

# Ajustar layout e mostrar
plt.tight_layout()
plt.show()
