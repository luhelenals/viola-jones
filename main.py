import cv2
import matplotlib.pyplot as plt

# Carregar o classificador pré-treinado (pode ser treinado manualmente também)
cascade = cv2.CascadeClassifier('C:/Users/luiza/source/repos/Viola Jones/cars.xml')

# Carregar imagem de teste
imagem = cv2.imread('C:/Users/luiza/source/repos/Viola Jones/cars.jpg')
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Detectar veículos
veiculos = cascade.detectMultiScale(
    cinza,
    scaleFactor=1.05,    # Tenta detectar melhor objetos pequenos
    minNeighbors=3,      # Aceita menos vizinhos para validar um objeto
    minSize=(50, 50),    # Ajuste para detectar carros menores
    maxSize=(400, 400)   # Ajuste para evitar detectar objetos muito grandes
)

# Desenhar retângulos nos veículos detectados
for (x, y, w, h) in veiculos:
    cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Exibir resultado
plt.figure(figsize=(5, 5))
plt.subplot(1, 1, 1)
plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()