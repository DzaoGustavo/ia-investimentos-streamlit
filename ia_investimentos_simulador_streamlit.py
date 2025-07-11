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
            text-align: center;
        }
        .logo-img {
            width: 180px;
            margin-bottom: 10px;
        }
        footer {
            text-align: center;
            font-size: 0.8em;
            color: #888;
            margin-top: 50px;
        }
        footer a {
            color: #ccc;
            text-decoration: none;
            margin: 0 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# LOGO E TÍTULO
st.markdown(
    """
    <div class="logo-title">
        <img class="logo-img" src="https://raw.githubusercontent.com/DzaoGustavo/ia-investimentos-streamlit/main/DZtech_Final.png" alt="Logo DzTech Invest AI">
        <h1>📊 DzTech Invest AI</h1>
        <p>Simulador Inteligente de Investimentos</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("Escolha um ativo, rode a IA e veja se ela compraria ou venderia com base na previsão!")

# LISTA DE ATIVOS
ativos = {
    "P
