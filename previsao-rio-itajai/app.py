
import streamlit as st
import joblib
import numpy as np
import json

st.set_page_config(page_title='Previsão do Nível do Rio Itajaí', layout='centered')

st.title("Previsão do Nível do Rio Itajaí")
st.markdown("""
Esta aplicação utiliza um modelo de **regressão linear** treinado com dados reais de chuva
e nível do rio em Rio do Sul para prever o nível do rio com base nas condições climáticas.
""")

try:
    modelo = joblib.load('modelo_rio_final.pkl')
    scaler = joblib.load('scaler_final.pkl')

    with open('info_modelo.json', 'r') as f:
        info = json.load(f)

    features = info['features_utilizadas']

    st.subheader("Insira os dados climáticos:")

    valores = []
    for feat in features:
        valor = st.number_input(f"{feat}:", format="%.2f")
        valores.append(valor)

    if st.button("Prever Nível do Rio"):
        entrada = np.array(valores).reshape(1, -1)
        entrada_scaled = scaler.transform(entrada)
        previsao = modelo.predict(entrada_scaled)[0]

        st.success(f"📈 Previsão: {previsao:.2f} centimetros")
        st.caption(f"Modelo: {info['melhor_modelo']} — R²: {info['metricas']['R2']:.4f}")

except Exception as e:
    st.error("Erro ao carregar os arquivos do modelo.")
    st.exception(e)
