#teste

# import yfinance as yf
# import plotly.graph_objects as go

# def plot_candlestick_com_previsao(ticker):
#     # Baixar os dados da ação
#     dados = yf.download(ticker, period="1mo")

#     # Renomear colunas para evitar problemas com Plotly
#     dados.columns = ["Open", "High", "Low", "Close", "Volume"]

#     print(dados.head())  # Verificar se os dados estão corretos

#     # Criar o gráfico de velas
#     fig = go.Figure(data=[go.Candlestick(
#         x=dados.index,
#         open=dados["Open"],
#         high=dados["High"],
#         low=dados["Low"],
#         close=dados["Close"]
#     )])

#     fig.update_layout(title=f"Gráfico de Candlestick - {ticker}")
#     fig.show()

# # Chamando a função
# plot_candlestick_com_previsao("COGN3.SA")

###############################################################################################

import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from modelo_rede_neural import preprocessar_dados, criar_modelo  # Importa funções do modelo treinado
import tensorflow as tf

def plot_candlestick_com_previsao(ticker="COGN3.SA", start="2020-01-01", end="2025-12-31"):
    # 🔹 Baixar dados históricos
    dados = yf.download(ticker, start=start, end=end)
    
    # Renomear colunas para evitar problemas com Plotly
    dados.columns = ["Open", "High", "Low", "Close", "Volume"]

    if dados.empty:
        print("Erro: Nenhum dado encontrado para o ativo.")
        return
    
    # 🔹 Preprocessar os dados
    X, _, scaler = preprocessar_dados(dados)
    
    # 🔹 Carregar o modelo treinado
    modelo = tf.keras.models.load_model("modelos/modelo.h5")

    # 🔹 Fazer a previsão para o próximo dia
    previsao_normalizada = modelo.predict(np.array([X[-1]]))
    previsao_real = scaler.inverse_transform(previsao_normalizada.reshape(-1, 1))[0][0]

    # 🔹 Adicionar a previsão como um novo ponto
    proximo_dia = pd.to_datetime(dados.index[-1]) + pd.Timedelta(days=1)
    dados.loc[proximo_dia] = [np.nan] * (len(dados.columns) - 1) + [previsao_real]

    # 🔹 Criar gráfico de Candlestick
    fig = go.Figure(data=[go.Candlestick(
        x=dados.index,
        open=dados["Open"],
        high=dados["High"],
        low=dados["Low"],
        close=dados["Close"],
        name="Histórico"
    )])

    # 🔹 Adicionar ponto previsto
    fig.add_trace(go.Scatter(
        x=[proximo_dia],
        y=[previsao_real],
        mode="markers",
        marker=dict(color="red", size=10),
        name="Previsão"
    ))

    # 🔹 Configuração do layout
    fig.update_layout(title=f"Gráfico de Candlestick para {ticker} com previsão",
                      xaxis_title="Data",
                      yaxis_title="Preço",
                      xaxis_rangeslider_visible=False)
    
    fig.show()

# 🔹 Executar para testar
plot_candlestick_com_previsao("COGN3.SA")



