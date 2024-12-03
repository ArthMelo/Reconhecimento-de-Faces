import cv2
import numpy as np
import os

# Função para pré-processar as imagens
def preprocessar_imagem(imagem, tamanho_imagem):
    cascata_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cascata_face.detectMultiScale(imagem, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        x, y, largura, altura = faces[0]
        imagem = imagem[y:y+altura, x:x+largura]
    imagem = cv2.resize(imagem, tamanho_imagem)
    imagem = cv2.equalizeHist(imagem)
    return imagem

# Função para carregar e processar as imagens
def carregar_imagens(pasta, tamanho_imagem=(50, 50)):
    imagens = []
    rotulos = []
    rotulo = 0
    for pasta_pessoa in os.listdir(pasta):
        caminho_pessoa = os.path.join(pasta, pasta_pessoa)
        if os.path.isdir(caminho_pessoa):
            for nome_arquivo in os.listdir(caminho_pessoa):
                caminho_arquivo = os.path.join(caminho_pessoa, nome_arquivo)
                imagem = cv2.imread(caminho_arquivo, cv2.IMREAD_GRAYSCALE)
                if imagem is not None:
                    imagem_processada = preprocessar_imagem(imagem, tamanho_imagem)
                    imagens.append(imagem_processada.flatten())
                    rotulos.append(rotulo)
            rotulo += 1
    return np.array(imagens), np.array(rotulos)

# Implementação manual do PCA
def pca(dados, retencao_variancia=0.95):
    vetor_medio = np.mean(dados, axis=0)
    dados_centralizados = dados - vetor_medio

    matriz_covariancia = np.cov(dados_centralizados, rowvar=False)
    autovalores, autovetores = np.linalg.eigh(matriz_covariancia)

    indices_ordenados = np.argsort(autovalores)[::-1]
    autovalores = autovalores[indices_ordenados]
    autovetores = autovetores[:, indices_ordenados]

    variancia_total = np.sum(autovalores)
    variancia_explicada = np.cumsum(autovalores) / variancia_total
    componentes_necessarios = np.searchsorted(variancia_explicada, retencao_variancia) + 1

    componentes_principais = autovetores[:, :componentes_necessarios]
    dados_reduzidos = np.dot(dados_centralizados, componentes_principais)

    return dados_reduzidos, componentes_principais, vetor_medio

# Função para reconhecimento facial
def reconhecer_face(imagens_treino, rotulos_treino, imagem_teste, tamanho_imagem=(50, 50)):
    imagem_teste_processada = preprocessar_imagem(imagem_teste, tamanho_imagem)

    # PCA
    treino_pca, componentes, vetor_medio = pca(imagens_treino, retencao_variancia=0.95)
    imagem_teste_centralizada = imagem_teste_processada.flatten() - vetor_medio
    teste_pca = np.dot(imagem_teste_centralizada, componentes)

    # Classificador k-NN simplificado
    distancias = np.linalg.norm(treino_pca - teste_pca, axis=1)
    indice_vizinho_mais_proximo = np.argmin(distancias)
    rotulo_predito = rotulos_treino[indice_vizinho_mais_proximo]
    distancia_relativa = distancias[indice_vizinho_mais_proximo]

    return rotulo_predito, distancia_relativa

# Mapeamento de rótulos para nomes
rotulo_para_nome = {
    0: "Ekko",
    1: "Jayce",
    2: "Jinx",
    3: "Vi"
}

# Carregar imagens de treino
pasta_faces = "faces_database"
imagens_treino, rotulos_treino = carregar_imagens(pasta_faces)

# Exemplo de uma imagem de teste
caminho_imagem_teste = "test_face.jpg"
imagem_teste = cv2.imread(caminho_imagem_teste, cv2.IMREAD_GRAYSCALE)
if imagem_teste is None:
    raise FileNotFoundError(f"Não foi possível carregar a imagem de teste: {caminho_imagem_teste}")

# Realizar o reconhecimento facial
rotulo_predito, distancia_relativa = reconhecer_face(imagens_treino, rotulos_treino, imagem_teste)
nome_predito = rotulo_para_nome.get(rotulo_predito, "Desconhecido")

# Exibir o resultado no terminal
print(f"Pessoa identificada: {nome_predito}")
print(f"Distância relativa: {distancia_relativa:.2f}")
