import numpy as np
import pandas as pd
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def preprocessar_dados(dados, tickers, look_back=120, salvar_scaler=True):
    """
    Prepara os dados para múltiplos ativos, mantendo a normalização separada para cada um.
    Retorna um dicionário com dados normalizados, escaladores e os conjuntos de treinamento.
    """
    if "Close" not in dados.columns.levels[0]:
        raise ValueError("Dados inválidos! Certifique-se de que contêm a coluna 'Close'.")
    
    os.makedirs("scalers", exist_ok=True)  # Criar diretório para scalers
    dados_processados = {}
    
    for ticker in tickers:
        preco_fechamento = dados[("Close", ticker)].dropna().values.reshape(-1, 1)

        if len(preco_fechamento) < look_back:
            print(f"Aviso: Dados insuficientes para {ticker}. Ignorando este ativo.")
            continue

        scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
        preco_fechamento_normalizado = scaler.fit_transform(preco_fechamento)
        
        if salvar_scaler:
            joblib.dump(scaler, f"scalers/scaler_{ticker}.pkl")  # Salvar o scaler

        X, y = [], []
        for i in range(look_back, len(preco_fechamento_normalizado)):
            X.append(preco_fechamento_normalizado[i-look_back:i, 0])
            y.append(preco_fechamento_normalizado[i, 0])

        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        dados_processados[ticker] = {
            "X": X, "y": y, "scaler": scaler
        }

    return dados_processados

def criar_modelo(input_shape, unidades=100, dropout_rate=0.1, otimizador="adam", loss="mean_squared_error"):
    """
    Cria um modelo LSTM configurável.
    """
    modelo = Sequential([
        LSTM(units=unidades, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units=unidades, return_sequences=False),
        Dropout(dropout_rate),
        Dense(units=1)
    ])
    
    modelo.compile(optimizer=otimizador, loss=loss)
    return modelo

def treinar_modelo(modelo, X, y, epochs=10, batch_size=32, salvar_modelo=True, nome_modelo="modelo.h5"):
    """
    Treina o modelo com callbacks para evitar overfitting.
    """
    if len(y) == 0:
        raise ValueError("Dados insuficientes para treinamento!")
    
    callbacks = [
        EarlyStopping(monitor='loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(nome_modelo, save_best_only=True, monitor='loss')
    ]
    
    modelo.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=callbacks)
    
    if salvar_modelo:
        modelo.save(nome_modelo)
        print(f"Modelo salvo como {nome_modelo}")
    
    return modelo

def carregar_modelo(nome_modelo):
    """
    Carrega um modelo treinado.
    """
    if not os.path.exists(nome_modelo):
        raise FileNotFoundError(f"O modelo {nome_modelo} não foi encontrado!")
    return load_model(nome_modelo)

def carregar_scaler(ticker):
    """
    Carrega o scaler salvo para um ativo específico.
    """
    caminho_scaler = f"scalers/scaler_{ticker}.pkl"
    if not os.path.exists(caminho_scaler):
        raise FileNotFoundError(f"Scaler para {ticker} não encontrado!")
    return joblib.load(caminho_scaler)
