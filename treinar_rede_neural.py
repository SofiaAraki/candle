# import os
# import pandas as pd
# import yfinance as yf
# from modelo_rede_neural import preprocessar_dados, criar_modelo

# # Lista de ativos a serem treinados
# tickers = ["COGN3.SA", "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA", "BBAS3.SA", "EMBR3.SA", "ITSA4.SA", "GOLL4.SA"]

# # Criar pasta para salvar os modelos, se não existir
# os.makedirs("modelos", exist_ok=True)

# # Baixar dados históricos
# dados = yf.download(tickers, start="2020-01-01", end="2025-12-31")

# # Processar dados para múltiplos ativos
# dados_processados = preprocessar_dados(dados, tickers)

# for ticker, dados_treino in dados_processados.items():
#     print(f"\nTreinando modelo para {ticker}...")

#     X, y = dados_treino["X"], dados_treino["y"]

#     modelo = criar_modelo(input_shape=(X.shape[1], 1))

#     modelo.fit(X, y, epochs=10, batch_size=32, verbose=1)

#     # Salvar o modelo treinado
#     modelo.save(f"modelos/modelo_{ticker}.h5")
#     print(f"Modelo para {ticker} salvo com sucesso!")

# print("\nTreinamento concluído para todos os ativos!")


#############################################################################
import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Lista de ativos a serem incluídos no modelo
tickers = ["COGN3.SA", "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ROXO34.SA", "BBAS3.SA", "EMBR3.SA", "ITSA4.SA", "GOLL4.SA"]

# Criar diretórios para salvar modelos e scalers
os.makedirs("modelos", exist_ok=True)
os.makedirs("scalers", exist_ok=True)

# Baixar os dados históricos
dados = yf.download(tickers, start="2020-01-01", end="2025-12-31")

# Definir qual ativo queremos prever
ativo_alvo = "ROXO34.SA"

def preprocessar_dados(dados, ativo_alvo, tickers, look_back=120):
    if 'Close' not in dados or dados.empty:
        raise ValueError("Dados inválidos ou vazios!")
    
    # Seleciona apenas os preços de fechamento de todos os ativos
    dados_fechamento = dados['Close'].dropna()
    
    # Verificação de dados ausentes
    if dados_fechamento.isna().sum().sum() > 0:
        print("Aviso: Dados ausentes encontrados! Preenchendo com valores anteriores...")
        dados_fechamento.fillna(method='ffill', inplace=True)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    dados_normalizados = scaler.fit_transform(dados_fechamento)
    
    joblib.dump(scaler, f"scalers/scaler_{ativo_alvo}.pkl")  # Salvar scaler para previsões futuras
    
    X, y = [], []
    for i in range(look_back, len(dados_normalizados)):
        X.append(dados_normalizados[i-look_back:i])
        y.append(dados_normalizados[i, tickers.index(ativo_alvo)])
    
    return np.array(X), np.array(y)

X, y = preprocessar_dados(dados, ativo_alvo, tickers)

def criar_modelo(input_shape):
    modelo = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=input_shape),
        Dropout(0.1),
        LSTM(units=100, return_sequences=False),
        Dropout(0.1),
        Dense(units=1)
    ])
    modelo.compile(optimizer='adam', loss='mean_squared_error')
    return modelo

modelo = criar_modelo((X.shape[1], X.shape[2]))

# Treinamento com Early Stopping para evitar overfitting
early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

print(f"Treinando modelo para prever {ativo_alvo} com base em múltiplos ativos...")
modelo.fit(X, y, epochs=50, batch_size=64, verbose=1, callbacks=[early_stopping])

# Salvar o modelo treinado
modelo.save(f"modelos/modelo_{ativo_alvo}.h5")
print(f"Modelo salvo com sucesso para {ativo_alvo}!")
