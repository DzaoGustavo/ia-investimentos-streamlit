import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.graph_objs as go

# CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="DzTech Invest AI", layout="centered", page_icon="📈")

# LOGO DA DZTECH CENTRALIZADO
st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <img src="https://raw.githubusercontent.com/DzaoGustavo/ia-investimentos-streamlit/main/DZtech_Final.png" alt="Logo DzTech" width="250">
    </div>
""", unsafe_allow_html=True)

# CSS PERSONALIZADO
st.markdown("""
    <style>
        body {
            background-color: #0f1117;
            color: #fafafa;
        }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            font-weight: bold;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #105fa0;
        }
    </style>
""", unsafe_allow_html=True)

# TÍTULO
st.title("😎 IA de Investimentos - Simulador Inteligente")
st.write("Escolha um ativo, rode a IA e veja se ela **compraria ou venderia** com base na previsão!")

# LISTA DE ATIVOS
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
    st.info(f"🔍 Coletando dados do ativo **{ticker}**...")

    df = yf.download(ticker, period="5y", interval="1d")
    df = df.dropna()

    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = df['Target']

    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.success(f"✅ Acurácia da IA: {accuracy * 100:.2f}%")

    # Previsão atual
    atual = X.iloc[-1].values.reshape(1, -1)
    previsao = model.predict(atual)[0]

    if previsao == 1:
        st.markdown("🟢 **A IA prevê que o preço vai subir.**")
        st.success(f"✅ Ordem simulada: **COMPRAR {ticker}**")
    else:
        st.markdown("🔴 **A IA prevê que o preço vai cair.**")
        st.error(f"❌ Ordem simulada: **VENDER {ticker}**")

    # Visualização do gráfico com erros e acertos
    df_pred = df[split:].copy()
    df_pred['Predito'] = y_pred
    df_pred['Correto'] = df_pred['Target'] == df_pred['Predito']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_pred.index,
        y=df_pred['Close'],
        mode='lines+markers',
        name='Preço',
        line=dict(color='white')
    ))
    fig.add_trace(go.Scatter(
        x=df_pred[df_pred['Correto']].index,
        y=df_pred[df_pred['Correto']]['Close'],
        mode='markers',
        name='Acertos',
        marker=dict(color='green', size=8, symbol='circle')
    ))
    fig.add_trace(go.Scatter(
        x=df_pred[~df_pred['Correto']].index,
        y=df_pred[~df_pred['Correto']]['Close'],
        mode='markers',
        name='Erros',
        marker=dict(color='red', size=8, symbol='x')
    ))

    fig.update_layout(
        title=f"📊 Previsões da IA para {ticker}",
        xaxis_title="Data",
        yaxis_title="Preço de Fechamento",
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="white")
    )

    st.plotly_chart(fig)

# RODAPÉ
st.markdown("""---""")
st.markdown(
    "<p style='text-align: center; font-size: 12px;'>© 2025 DzTech Invest AI — Links profissionais em breve.</p>",
    unsafe_allow_html=True
)
