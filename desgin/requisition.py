import requests

def enviar_requisicao():
    # Receber os dados do formulário
    dados_entrada = requests.args.to_dict()

    # Criar uma nova instância do modelo
    with open("model.pkl", "rb") as f:
        modelo = pickle.load(f)

    # Fazer uma previsão
    predicao = modelo.predict(list(dados_entrada.values()))

    # Imprimir a previsão
    print(f"Previsão: {predicao}")

if __name__ == "__main__":
    enviar_requisicao()
