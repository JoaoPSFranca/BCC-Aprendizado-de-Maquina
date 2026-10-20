# 1) Carregar os dados BRUTOS (antes de qualquer tratamento estatístico)
dados = pd.read_csv("heart.csv")

# 2) Tratar só o que NÃO depende de estatística nenhuma
#    (ex.: Cholesterol == 0 não faz sentido -> vira NaN; isso é regra de negócio, não estatística)
dados['Cholesterol'] = dados['Cholesterol'].replace(0, np.nan)

# 3) SEPARAR treino/teste AGORA, com os NaN ainda presentes
treino, teste = train_test_split(dados, test_size=0.3, random_state=0)

# 4) Aprender a média (e qualquer outra estatística) SÓ no treino
media_idade_treino = treino['Age'].mean()
media_colesterol_treino = treino['Cholesterol'].mean()

# 5) Aplicar (transform) essa MESMA média nos dois lados - ninguém fica sem tratamento
treino['Age']        = treino['Age'].fillna(media_idade_treino)
teste['Age']         = teste['Age'].fillna(media_idade_treino)
treino['Cholesterol'] = treino['Cholesterol'].fillna(media_colesterol_treino)
teste['Cholesterol']  = teste['Cholesterol'].fillna(media_colesterol_treino)

# 6) Encoders (LabelEncoder/OneHotEncoder): fit só no treino, transform nos dois
# 7) StandardScaler: fit_transform no treino, só transform no teste
scaler = StandardScaler()
x_treino_esc = scaler.fit_transform(treino[colunas])
x_teste_esc  = scaler.transform(teste[colunas])   # nunca fit aqui