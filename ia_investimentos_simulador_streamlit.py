import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.graph_objs as go

# CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="DzTech Invest AI", layout="centered", page_icon="📈")

# CSS PERSONALIZADO
st.markdown(
    """
    <style>
        body {
            background-color: #0f1117;
            color: #fafafa;
        }
        .logo-title {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1.5rem;
        }
        .logo-title img {
            width: 90px;
            margin-right: 1rem;
            border-radius: 8px;
        }
        .logo-title h1 {
            font-size: 2.5rem;
            font-weight: 800;
        }
        .footer {
            text-align: center;
            font-size: 0.8rem;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #333;
            color: #888;
        }
        .footer a {
            color: #bbb;
            text-decoration: none;
            margin: 0 10px;
        }
    </style>
    """, unsafe_allow_html=True
)

# LOGO + TÍTULO
st.markdown('''
<div class="logo-title">
    <img src="https://raw.githubusercontent.com/seuusuario/seurepo/main/DZtech_Final.png">
    <h1>DzTech Invest AI</h1>
</div>
''', unsafe_allow_html=True)

st.markdown("🚀 IA que simula decisões de investimento com base no histórico de preços. Selecione um ativo, rode a IA e veja a previsão!")

# INICIALIZA HISTÓRICO NA SESSÃO
if "historico" not in st.session_state:
    st.session_state.historico = []

# LISTA DE ATIVOS DISPONÍVEIS
ativos = {
    "Petrobras (PETR4)": "PETR4.SA",
    "Vale (VALE3)": "VALE3.SA",
    "Assaí (ASAI3)": "ASAI3.SA",
    "Apple (AAPL)": "AAPL",
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Chainlink (LINK)": "LINK-USD"
}

ativo_nome = st.selectbox("📌 Selecione o ativo:", list(ativos.keys()))
ticker = ativos[ativo_nome]

# BOTÃO DE EXECUÇÃO
if st.button("📊 Rodar IA"):
    df = yf.download(ticker, start='2019-01-01', end='2024-07-01')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)

    # FEATURES + TARGET
    df['Return'] = df['Close'].pct_change()
    df['Target'] = np.where(df['Return'].shift(-1) > 0, 1, 0)
    df['Volatility'] = df['Return'].rolling(window=5).std()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df.dropna(inplace=True)

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

    st.success(f"Acurácia do modelo: {accuracy:.2%}")

    ultima_previsao = y_pred[-1]
    data_analise = df.index[-1].strftime("%Y-%m-%d")
    decisao = "COMPRAR" if ultima_previsao == 1 else "VENDER"

    if ultima_previsao == 1:
        st.markdown("🟢 **A IA prevê alta.**")
        st.success(f"✅ Ordem simulada: COMPRAR **{ticker}**")
    else:
        st.markdown("🔴 **A IA prevê queda.**")
        st.error(f"🚫 Ordem simulada: VENDER **{ticker}**")

    st.session_state.historico.append({
        "Data": data_analise,
        "Ativo": ticker,
        "Decisão IA": decisao,
        "Acurácia do Modelo": f"{accuracy:.2%}"
    })

    # GRÁFICO INTERATIVO
    df_test = df[split:].copy()
    df_test['Predicted'] = y_pred
    df_test['Correct'] = df_test['Target'] == df_test['Predicted']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_test.index, y=df_test['Close'], mode='lines', name='Preço', line=dict(color='lightblue')))
    fig.add_trace(go.Scatter(x=df_test.index[df_test['Correct']], y=df_test['Close'][df_test['Correct']],
                             mode='markers', name='Acertos', marker=dict(color='green', symbol='circle')))
    fig.add_trace(go.Scatter(x=df_test.index[~df_test['Correct']], y=df_test['Close'][~df_test['Correct']],
                             mode='markers', name='Erros', marker=dict(color='red', symbol='x')))

    fig.update_layout(title=f"Gráfico de Preço + Predições IA ({ticker})",
                      xaxis_title='Data',
                      yaxis_title='Preço',
                      template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

# HISTÓRICO DE EXECUÇÕES
if st.session_state.historico:
    st.subheader("📜 Histórico de Previsões")
    df_hist = pd.DataFrame(st.session_state.historico)
    st.dataframe(df_hist, use_container_width=True)

    csv = df_hist.to_csv(index=False)
    st.download_button("📥 Exportar Histórico CSV", csv, "historico_dztech_ai.csv", "text/csv")

# RODAPÉ COM LINKS (PLACEHOLDERS)
st.markdown("""
<div class="footer">
    DzTech Invest AI · Desenvolvido por Dzão<br>
    <a href="https://github.com/seu-github" target="_blank">GitHub</a> | 
    <a href="https://linkedin.com/in/seu-linkedin" target="_blank">LinkedIn</a> | 
    <a href="https://seu-site.com" target="_blank">Portfólio</a>
</div>
""", unsafe_allow_html=True)
