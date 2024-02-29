import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Caminhos para o modelo e o conjunto de testes
modelo_path = 'C:/Users/lucas/OneDrive/Área de Trabalho/Sistemas Inteligentes/ProjetoFinal_SI/modelo_flor50.keras'
conjunto_teste_path = 'C:/Users/lucas/OneDrive/Área de Trabalho/Sistemas Inteligentes/ProjetoFinal_SI/teste'

# Carregando o modelo treinado
modelo = load_model(modelo_path)

# Parâmetros
batch_size = 32
num_testes = 5

# Criando gerador de imagens para o conjunto de testes
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    conjunto_teste_path,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Inicializando a matriz de confusão acumulada
conf_matrix_sum = np.zeros((len(test_generator.class_indices), len(test_generator.class_indices)))

# Lista para armazenar a acurácia de cada classe em cada teste
class_accuracies = [[] for _ in range(len(test_generator.class_indices))]

# Loop para realizar os testes múltiplos
for teste in range(num_testes):
    # Realizando a predição no conjunto de testes
    predictions = modelo.predict(test_generator, steps=len(test_generator), verbose=1)
    
    # Convertendo as probabilidades para rótulos
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = test_generator.classes
    
    # Calculando a matriz de confusão para este teste
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    
    # Acumulando a matriz de confusão
    conf_matrix_sum += conf_matrix
    
    # Calculando a acurácia para cada classe
    class_accuracy = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    
    # Armazenando as acurácias de cada classe
    for i, acc in enumerate(class_accuracy):
        class_accuracies[i].append(acc)

# Calculando a média da matriz de confusão
conf_matrix_avg = conf_matrix_sum / num_testes

# Calculando a média da acurácia por classe
accuracy_means = np.mean(class_accuracies, axis=1)

# Plotando o gráfico de dispersão para a média de acurácia por classe
plt.figure(figsize=(10, 6))
plt.scatter(range(len(test_generator.class_indices)), accuracy_means, color='blue', marker='o')
plt.xticks(ticks=range(len(test_generator.class_indices)), labels=test_generator.class_indices.keys(), rotation=45)
plt.xlabel('Classe')
plt.ylabel('Acurácia Média')
plt.title('Média de Acurácia por Classe')
plt.grid(False)
plt.show()

# Plotando a matriz de confusão média usando seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_avg, annot=True, fmt=".2f", cmap="Blues", xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.xlabel('Rótulos Preditos')
plt.ylabel('Rótulos Verdadeiros')
plt.title('Matriz de Confusão Média')
plt.show()
