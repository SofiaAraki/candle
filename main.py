import yfinance as yf
import plotly.graph_objects as go
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler

def carregar_dados(tickers):
    """
    Baixa os dados históricos de múltiplos ativos e retorna um DataFrame.
    """
    dados = yf.download(tickers, period="1y")
    return dados

def normalizar_dados(dados, tickers):
    """
    Normaliza os dados de fechamento de todos os ativos.
    """
    dados_fechamento = dados['Close'].dropna()
    scaler = MinMaxScaler(feature_range=(0, 1))
    dados_normalizados = scaler.fit_transform(dados_fechamento)
    return dados_normalizados, scaler

def prever_fechamento(modelo, dados, ticker, scaler, tickers):
    """
    Faz a previsão do fechamento para um ativo específico.
    """
    dados_normalizados, _ = normalizar_dados(dados, tickers)
    ultimos_precos = np.array(dados_normalizados[-120:]).reshape(1, 120, -1)
    previsao_normalizada = modelo.predict(ultimos_precos)[0][0]
    previsao_real = scaler.inverse_transform([[previsao_normalizada] * len(tickers)])[0][tickers.index(ticker)]
    return previsao_real

def plot_candlestick_com_previsao(dados, ticker, previsao):
    """
    Plota um gráfico de candlestick do ativo selecionado e adiciona a previsão.
    """
    fig = go.Figure(data=[go.Candlestick(
        x=dados.index,
        open=dados[('Open', ticker)],
        high=dados[('High', ticker)],
        low=dados[('Low', ticker)],
        close=dados[('Close', ticker)]
    )])

    # Adicionando ponto de previsão
    proxima_data = dados.index[-1] + datetime.timedelta(days=1)
    fig.add_trace(go.Scatter(
        x=[proxima_data],
        y=[previsao],
        mode="markers",
        marker=dict(color="red", size=10),
        name="Previsão"
    ))

    fig.update_layout(title=f"Gráfico de Candlestick - {ticker} com Previsão")
    fig.show()

# Lista de ativos a serem baixados
tickers = ["COGN3.SA", "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ROXO34.SA", "BBAS3.SA", "EMBR3.SA", "ITSA4.SA", "GOLL4.SA"]

# Ativo que queremos prever
ticker_para_prever = "PETR4.SA"

# Carregar os dados de múltiplos ativos
dados = carregar_dados(tickers)

# Carregar modelo treinado
modelo = tf.keras.models.load_model(f"modelos/modelo_{ticker_para_prever}.h5")

# Normalizar os dados e obter o scaler
dados_normalizados, scaler = normalizar_dados(dados, tickers)

# Fazer previsão para o ativo escolhido
previsao = prever_fechamento(modelo, dados, ticker_para_prever, scaler, tickers)
print(f"Previsão para o próximo fechamento de {ticker_para_prever}: {previsao:.2f}")

# Plotar gráfico apenas para o ativo escolhido
plot_candlestick_com_previsao(dados, ticker_para_prever, previsao)
