import cv2
import numpy as np
from sklearn.decomposition import PCA
import os

# Função para carregar as imagens de um diretório e converter para vetores
def load_images_from_folder(folder, image_size=(100, 100)):
    images = []
    labels = []
    label = 0
    for person_folder in os.listdir(folder):
        person_path = os.path.join(folder, person_folder)
        if os.path.isdir(person_path):
            for filename in os.listdir(person_path):
                img_path = os.path.join(person_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Redimensionar a imagem para garantir que todas tenham o mesmo tamanho
                    img_resized = cv2.resize(img, image_size)
                    images.append(img_resized.flatten())  # "Achatar" a imagem em um vetor 1D
                    labels.append(label)
            label += 1
    return np.array(images), np.array(labels)

# Função para realizar o reconhecimento facial usando PCA
def recognize_face(train_images, train_labels, test_image, n_components=None, image_size=(100, 100)):
    # Redimensionar a imagem de teste para o mesmo tamanho das imagens de treinamento
    test_image_resized = cv2.resize(test_image, image_size)

    # Definir o número de componentes principais para o PCA
    n_samples, n_features = train_images.shape
    if n_components is None or n_components > n_samples or n_components > n_features:
        n_components = min(n_samples, n_features)  # Ajustar para o número máximo possível de componentes

    # Aplica o PCA nos dados de treino
    pca = PCA(n_components=n_components)
    pca.fit(train_images)

    # Projeção dos dados de treino nos componentes principais
    train_pca = pca.transform(train_images)

    # Projeção da imagem de teste nos componentes principais
    test_image_pca = pca.transform([test_image_resized.flatten()])

    # Calcula as distâncias Euclidianas entre a imagem de teste e as imagens de treino
    distances = np.linalg.norm(train_pca - test_image_pca, axis=1)

    # Identifica o rótulo da imagem mais próxima
    closest_match = np.argmin(distances)
    return train_labels[closest_match]

# Mapeamento de Rótulos para Nomes
label_to_name = {
    0: "Alan",
    1: "Jim",
    2: "Jinx",
    3: "Neymar"
}

# Carregar imagens de treino
folder = "faces_database"  # Caminho do diretório com as imagens de treino
train_images, train_labels = load_images_from_folder(folder)

# Exemplo de uma imagem de teste
test_img_path = "test_face.jpg"  # Caminho de uma imagem de teste
test_image = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)

# Verificar se a imagem de teste foi carregada corretamente
if test_image is None:
    raise FileNotFoundError(f"Não foi possível carregar a imagem de teste: {test_img_path}")

# Realizar o reconhecimento facial
predicted_label = recognize_face(train_images, train_labels, test_image)

# Converter o rótulo previsto para o nome
predicted_name = label_to_name.get(predicted_label, "Desconhecido")

# Adicionar o nome na imagem de teste
test_image_color = cv2.imread(test_img_path)  # Carregar imagem colorida para exibição
if test_image_color is not None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(test_image_color, predicted_name, (10, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Exibir a imagem com o nome
    cv2.imshow("Imagem de Teste", test_image_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"A pessoa identificada é: {predicted_name}")
