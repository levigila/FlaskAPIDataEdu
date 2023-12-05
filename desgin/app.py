from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

data = pd.read_csv('Desempenho_alunos.csv', index_col=0)

modelo = joblib.load('modelo_svm.pkl')

app = Flask(__name__)
app.url_prefix = '/requisicao'

@app.route('/', methods=['POST'])
def prever():
    try:
        dados_entrada = request.get_json()

        # Validar dados de entrada
        if not dados_entrada:
            raise Exception('Dados de entrada não fornecidos')
        if not all(k in dados_entrada for k in ['Idade', 'Salario_Total', 'Media_Acumulada_Ultimo_Periodo']):
            raise Exception('Dados de entrada incompletos')

        # Validar idade
        if not dados_entrada['Idade'].isdigit():
            raise Exception('Idade deve ser um número')

        # Validar salário total
        if not dados_entrada['Salario_Total'].isdigit():
            raise Exception('Salário total deve ser um número')

        # Validar média acumulada do último período
        if not dados_entrada['Media_Acumulada_Ultimo_Periodo'].isdigit():
            raise Exception('Média acumulada do último período deve ser um número')

        # Convertir idade para um valor inteiro
        dados_entrada['Idade'] = int(dados_entrada['Idade'])

        # Convertir salário total para um valor float
        dados_entrada['Salario_Total'] = float(dados_entrada['Salario_Total'])

        # Convertir média acumulada do último período para um valor float
        dados_entrada['Media_Acumulada_Ultimo_Periodo'] = float(dados_entrada['Media_Acumulada_Ultimo_Periodo'])

        # Criar um DataFrame a partir dos dados de entrada do aluno
        df_entrada = pd.DataFrame([dados_entrada])

        # Converter variáveis categóricas usando pd.get_dummies
        df_entrada_encoded = pd.get_dummies(df_entrada)

        # Mapear valores para os esperados pelo modelo
        for col, mapping in zip(df_entrada_encoded.columns):
            df_entrada_encoded[col] = df_entrada_encoded[col].map(mapping)

        # Definir um limite para a pontuação de previsão
        limite_previsao = 0.5

        # Fazer a previsão usando o modelo
        previsao = modelo.predict(df_entrada_encoded)

        # Definir o resultado da previsão
        if previsao > limite_previsao:
            resultado['previsao'] = 'Aprovado'
        else:
            resultado['previsao'] = 'Reprovado'

        return jsonify(resultado)
    except Exception as e:
        print(f"Erro no servidor Flask: {e}")
        return jsonify({'erro': 'Erro interno no servidor'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
