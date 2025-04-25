# 📈 Previsão de Preços de Ações com LSTM e Dados Multivariados

Este projeto tem como objetivo prever o preço de fechamento de um ativo da bolsa brasileira utilizando redes neurais LSTM (Long Short-Term Memory) e dados históricos de múltiplos ativos como entrada. A previsão é feita com base em séries temporais de 120 dias.

---

## 🔍 Objetivo

Prever o próximo preço de fechamento de um ativo específico, utilizando não só seu próprio histórico, mas também o histórico de outros ativos do mercado como contexto.

---

## ⚙️ Tecnologias Utilizadas

- Python 3.x  
- TensorFlow / Keras  
- Scikit-learn  
- Pandas  
- NumPy  
- yFinance  
- Plotly (para visualização)  

---

## 🧠 Abordagem

1. **Coleta dos Dados**  
   Utiliza a API do [Yahoo Finance](https://finance.yahoo.com) via `yfinance` para baixar dados históricos de múltiplos ativos.

2. **Pré-processamento**  
   - Seleção apenas dos preços de **fechamento**.
   - Normalização dos dados com `MinMaxScaler`.
   - Geração de janelas de tempo (`look_back = 120`) para alimentar a LSTM.

3. **Treinamento do Modelo**  
   - Modelo LSTM com duas camadas e `Dropout` para evitar overfitting.
   - Saída única para prever apenas o ativo-alvo.
   - O modelo é treinado e salvo com o nome `modelo_<TICKER>.h5`.

4. **Previsão**  
   - O modelo faz a previsão com base nos últimos 120 dias de dados normalizados.
   - O valor previsto é **desnormalizado** para obter a cotação real.

5. **Visualização**  
   - Um gráfico **candlestick** do ativo é gerado com o preço previsto plotado como um ponto vermelho.

---

## 🧾 Exemplos de Ativos Utilizados

- COGN3.SA (Cogna)  
- PETR4.SA (Petrobras PN)  
- VALE3.SA (Vale)  
- ITUB4.SA (Itaú Unibanco)  
- BBDC4.SA (Bradesco)  
- ROXO34.SA (Nubank - BDR)  
- BBAS3.SA (Banco do Brasil)  
- EMBR3.SA (Embraer)  
- ITSA4.SA (Itaúsa)  
- GOLL4.SA (Gol Linhas Aéreas)

---

## 📂 Estrutura do Projeto

```
📦 candle/
│
├── modelos/
│   └── modelo_<TICKER>.h5         # Modelos treinados por ativo
│
├── treino.py                      # Script de treinamento
├── main.py                        # Script de previsão e visualização
├── requirements.txt               # Bibliotecas utilizadas
└── README.md                      # Documentação do projeto
```

---

## 🚀 Como Executar

1. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Treine um modelo:**
   ```bash
   python treino.py
   ```

3. **Faça a previsão com gráfico:**
   ```bash
   python main.py
   ```

---

## 📌 Observações

- O projeto utiliza **dados normalizados multivariados** para enriquecer a entrada da LSTM.
- Cada modelo é específico para um ativo, mesmo que os dados de entrada sejam os mesmos.

---

## 📈 Resultado

A saída será um gráfico candlestick com o valor previsto marcado em vermelho, além de uma previsão numérica exibida no console.

---

## 📬 Contribuições

Sugestões, melhorias e correções são sempre bem-vindas! Sinta-se à vontade para abrir uma _issue_ ou um _pull request_.

---

## 🧑‍💻 Desenvolvido por

**Sofia** – [LinkedIn](https://www.linkedin.com/in/sofiaaraki/) / [GitHub](https://github.com/SofiaAraki/)
