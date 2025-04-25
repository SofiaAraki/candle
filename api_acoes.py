from fastapi import FastAPI
import yfinance as yf

app = FastAPI()

@app.get("/cotacoes/{ticker}")
def get_cotacoes(ticker: str, start: str = "2020-01-01", end: str = "2025-12-31"):
    # Usando yf.Ticker() para obter os dados históricos
    stock = yf.Ticker(ticker)
    dados = stock.history(start=start, end=end)

    if dados.empty:
        return {"erro": "Nenhum dado encontrado"}

    # Resetando o índice e ajustando as colunas disponíveis
    dados_reset = dados.reset_index()

    # Se 'Adj Close' não estiver presente, podemos trabalhar com as colunas disponíveis
    colunas_necessarias = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    colunas_presentes = [col for col in colunas_necessarias if col in dados_reset.columns]
    
    dados_reset = dados_reset[colunas_presentes]

    # Convertendo para o formato desejado
    response = dados_reset.to_dict(orient="records")
    return {"data": response}

# #falta fazer tratamento de erro caso a ação não exista
# #criar outros endpionts para calcular outras métricas(featres) como: media movel, bollinger bands, etc
# #conferir as datas para o formato YYYY-MM-DD


