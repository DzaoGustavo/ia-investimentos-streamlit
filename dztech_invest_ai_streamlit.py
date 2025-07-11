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
st.write("Escolha um ativo, rode a IA e veja se ela **compr**
