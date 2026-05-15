import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px
import numpy as np

# 1. Dataset
gamer = pd.DataFrame({
    'horas_jogo': [1, 2, 4, 6, 8, 10],
    'cansaco': [1, 2, 3, 5, 8, 10]
})

# 2. Variável independente (X) e dependente (y)
X = gamer[['horas_jogo']]
y = gamer['cansaco']

# 3. Criação do modelo
modelo = LinearRegression()

# 4. Treinamento
modelo.fit(X, y)

# 5. Previsão
horas = [[7]]
previsao = modelo.predict(horas)

# 6. Resultado
print(f'Previsão de cansaço para 7 horas jogando: {previsao[0]:.2f}')

# -------------------------------
# REPRESENTAÇÃO DO GRÁFICO
# -------------------------------

# Criando valores para a linha de regressão
x_linha = np.linspace(1, 10, 100)
y_linha = modelo.predict(x_linha.reshape(-1, 1))

# Gráfico interativo
fig = px.scatter(
    gamer,
    x='horas_jogo',
    y='cansaco',
    title='Relação entre Horas de Jogo e Cansaço',
    labels={
        'horas_jogo': 'Horas Jogando',
        'cansaco': 'Nível de Cansaço'
    }
)

# Adicionando linha de regressão
fig.add_scatter(
    x=x_linha,
    y=y_linha,
    mode='lines',
    name='Linha de Regressão'
)

# Mostrar gráfico
fig.show()