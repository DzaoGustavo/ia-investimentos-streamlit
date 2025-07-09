
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="IA de Investimentos", layout="centered")

st.title("🤖 IA de Investimentos - Simulador Inteligente")
st.write("Escolha um ativo, rode a IA e veja se ela compraria ou venderia com base na previsão!")

# Seleção de ativo
ativos = {
    "Petrobras (PETR4)": "PETR4.SA",
    "Vale (VALE3)": "VALE3.SA",
    "Assaí (ASAI3)": "ASAI3.SA",
    "Apple (AAPL)": "AAPL",
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Chainlink (LINK)": "LINK-USD"
}

ativo_nome = st.selectbox("Selecione o ativo:", list(ativos.keys()))
ticker = ativos[ativo_nome]

if st.button("▶️ Rodar IA"):
    st.write(f"🔍 Coletando dados do ativo **{ticker}**...")

    df = yf.download(ticker, start='2019-01-01', end='2024-07-01')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)

    # Features
    df['Return'] = df['Close'].pct_change()
    df['Target'] = np.where(df['Return'].shift(-1) > 0, 1, 0)
    df['Volatility'] = df['Return'].rolling(window=5).std()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df.dropna(inplace=True)

    # Treino/teste
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Volatility', 'MA_5', 'MA_10']
    X = df[features]
    y = df['Target']
    split = int(0.8 * len(df))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.success(f"Acurácia da IA: {accuracy:.2%}")

    # Simulação de decisão
    ultima_previsao = y_pred[-1]
    if ultima_previsao == 1:
        st.markdown("🟢 **A IA prevê que o preço vai subir.**")
        st.success(f"✅ Ordem simulada: **COMPRAR {ticker}**")
    else:
        st.markdown("🔴 **A IA prevê que o preço vai cair.**")
        st.error(f"🚫 Ordem simulada: **VENDER {ticker}**")

    # Gráfico de acertos/erros
    df_test = df[split:].copy()
    df_test['Predicted'] = y_pred
    df_test['Correct'] = df_test['Target'] == df_test['Predicted']

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_test.index, df_test['Close'], label='Preço')
    ax.scatter(df_test.index[df_test['Correct']], df_test['Close'][df_test['Correct']], label='Acertos', color='green', marker='o')
    ax.scatter(df_test.index[~df_test['Correct']], df_test['Close'][~df_test['Correct']], label='Erros', color='red', marker='x')
    ax.set_title(f'Predições da IA para {ticker}')
    ax.set_xlabel('Data')
    ax.set_ylabel('Preço')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
