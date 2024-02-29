
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Diretório onde os dados estão localizados
data_dir = 'C:/Users/lucas/OneDrive/Área de Trabalho/Sistemas Inteligentes/ProjetoFinal_SI/archive/flowers'

# Definir o tamanho das imagens e o batch size
img_width, img_height = 150, 150
batch_size = 32

# Criar gerador de dados para treino e validação
datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Divisão em treino e validação
)

# Carregar e dividir os dados de treino e validação
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'categorical',
    subset = 'training',
    seed = 33
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'categorical',
    subset = 'validation',
    seed = 33
)

# Obtém os rótulos das classes e seus índices
class_labels = list(train_generator.class_indices.keys())

# Obtém a contagem de amostras por classe no conjunto de treino
train_class_counts = train_generator.classes
train_class_counts = np.bincount(train_class_counts)

# Obtém a contagem de amostras por classe no conjunto de validação
validation_class_counts = validation_generator.classes
validation_class_counts = np.bincount(validation_class_counts)

# Imprime a contagem de amostras por classe
print("Contagem de amostras por classe no conjunto de treino:")
for i, label in enumerate(class_labels):
    print(f"{label}: {train_class_counts[i]}")

print("\nContagem de amostras por classe no conjunto de validacao:")
for i, label in enumerate(class_labels):
    print(f"{label}: {validation_class_counts[i]}")

# Crie o modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), activation='relu'),
    MaxPooling2D(pool_size = (2, 2)),
    Conv2D(64, (3, 3), activation = 'relu'),
    MaxPooling2D(pool_size = (2, 2)),
    Conv2D(128, (3, 3), activation = 'relu'),
    MaxPooling2D(pool_size = (2, 2)),
    Flatten(),
    Dense(128, activation = 'relu'),
    Dense(len(train_generator.class_indices), activation='softmax')  # Número de classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treine o modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs = 50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Salve o modelo treinado
model.save('C:/Users/lucas/OneDrive/Área de Trabalho/ProjetoFinal_SI/modelo_flor.keras')

# Visualize as curvas de acurácia
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
