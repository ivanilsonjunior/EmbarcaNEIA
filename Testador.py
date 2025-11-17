import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import os

# ============================
# 1. CARREGAR MODELO TREINADO
# ============================
model = keras.models.load_model("Nordestino.keras")
print("Modelo carregado com sucesso!")


# ============================
# 2. FUNÇÃO PARA CARREGAR IMAGEM (URL OU ARQUIVO)
# ============================
def carregar_imagem(caminho_ou_url, image_size=(180, 180)):
    # Caso seja URL (começa com http)
    if caminho_ou_url.startswith("http://") or caminho_ou_url.startswith("https://"):
        print("Baixando imagem da URL...")
        resp = requests.get(caminho_ou_url)
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        # arquivo local
        if not os.path.exists(caminho_ou_url):
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho_ou_url}")
        img = Image.open(caminho_ou_url).convert("RGB")

    # Redimensiona para o tamanho do modelo
    img = img.resize(image_size)

    return img


# ============================
# 3. FUNÇÃO DE CLASSIFICAÇÃO
# ============================
def classificar_imagem(caminho_ou_url, model, image_size=(180, 180)):
    img = carregar_imagem(caminho_ou_url, image_size)

    # Mostrar imagem
    plt.imshow(img)
    plt.title(f"Imagem carregada")
    plt.axis("off")
    plt.show()

    # Converter para array
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # Predição
    logits = model.predict(img_array)[0][0]
    score = float(tf.sigmoid(logits))

    print("\n🔍 RESULTADO DA CLASSIFICAÇÃO")
    print(f"Logit: {logits:.4f}")
    print(f"Probabilidade de Tapioca: {100 * score:.2f}%")
    print(f"Probabilidade de Cuscuz:  {100 * (1 - score):.2f}%")

    if score > 0.5:
        print("👉 O modelo classifica como **Tapioca**")
    else:
        print("👉 O modelo classifica como **Cuscuz**")

    print("-" * 50)


# ============================
# 4. MODO INTERATIVO: PEDIR URL AO USUÁRIO
# ============================
while True:
    entrada = input("\nDigite a URL ou caminho da imagem (ou 'sair'): ")

    if entrada.lower() == "sair":
        break

    try:
        classificar_imagem(entrada, model)
    except Exception as e:
        print("Erro ao carregar ou processar a imagem:", e)
