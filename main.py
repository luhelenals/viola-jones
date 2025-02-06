import cv2
import matplotlib.pyplot as plt

def main():
    file = input('Nome do arquivo: ')
    cascade, cinza, imagem = load_image(file)
    draw_vehicles(imagem, cinza, cascade)
    show_image(imagem)

# Carregar imagem de teste
def load_image(file):
    # Carregar o classificador pré-treinado
    cascade = cv2.CascadeClassifier('cars.xml')

    imagem = cv2.imread(file)
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    return cascade, cinza, imagem

def draw_vehicles(src, cinza, cascade):
    # Detectar veículos
    veiculos = cascade.detectMultiScale(
        cinza,
        scaleFactor=1.05, 
        minNeighbors=3,   
        minSize=(50, 50), 
        maxSize=(400, 400)
    )

    # Desenhar retângulos nos veículos detectados
    for (x, y, w, h) in veiculos:
        cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Exibir resultado
def show_image(src):
    plt.figure(figsize=(5, 5))
    plt.subplot(1, 1, 1)
    plt.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

if __name__ == '__main__':
    main()