# ==========================================
# PREVISÃO DE NOTAS COM SCIKIT-LEARN
# STREAMLIT + PLOTLY
# ==========================================

# Bibliotecas
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.linear_model import LinearRegression

# ==========================================
# 1. CONJUNTO DE DADOS
# ==========================================

estudos = pd.DataFrame({
    'notas': [1, 2, 4, 6, 8, 10],
    'horas': [2, 4, 5, 7, 9, 10]
})

# ==========================================
# 2. VARIÁVEIS
# ==========================================

# Variável independente
X = estudos[['horas']]

# Variável dependente
y = estudos['notas']

# ==========================================
# 3. CRIAÇÃO DO MODELO
# ==========================================

modelo = LinearRegression()

# Treinamento do modelo
modelo.fit(X, y)

# ==========================================
# 4. INTERFACE STREAMLIT
# ==========================================

st.set_page_config(
    page_title="Previsão de Notas",
    layout="centered"
)

st.title("Previsão de Notas com IA")

st.write("""
Aplicação de Machine Learning utilizando
Regressão Linear com Scikit-learn.
""")

# ==========================================
# 5. ENTRADA DE DADOS
# ==========================================

horas_estudo = st.slider(
    "Horas de estudo",
    min_value=1,
    max_value=12,
    value=6
)

# ==========================================
# 6. PREVISÃO
# ==========================================

previsao = modelo.predict([[horas_estudo]])

st.subheader("Resultado da previsão")

st.success(
    f"Nota prevista: {previsao[0]:.2f}"
)

# ==========================================
# 7. INFORMAÇÕES DO MODELO
# ==========================================

st.subheader("Parâmetros do Modelo")

st.write(f"Coeficiente Angular: {modelo.coef_[0]:.2f}")
st.write(f"Intercepto: {modelo.intercept_:.2f}")

# ==========================================
# 8. LINHA DE REGRESSÃO
# ==========================================

# Gerando previsões
estudos['previsao'] = modelo.predict(X)

# ==========================================
# 9. GRÁFICO COM PLOTLY
# ==========================================

fig = px.scatter(
    estudos,
    x='horas',
    y='notas',
    title='Horas de Estudo x Notas',
    labels={
        'horas': 'Horas de Estudo',
        'notas': 'Notas'
    }
)

# Adiciona linha de regressão
fig.add_scatter(
    x=estudos['horas'],
    y=estudos['previsao'],
    mode='lines',
    name='Regressão Linear'
)

# Exibe gráfico
st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# 10. EXIBIÇÃO DOS DADOS
# ==========================================

st.subheader("Base de Dados")

st.dataframe(estudos)