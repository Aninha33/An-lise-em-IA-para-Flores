import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tkinter import ttk
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import ttk


# Diretório onde os dados estão localizados
data_dir = 'C:/Users/lucas/OneDrive/Área de Trabalho/Sistemas Inteligentes/ProjetoFinal_SI/archive/flowers'

# Definir o tamanho das imagens e o batch size
img_width, img_height = 150, 150
batch_size = 32

# Carregar o modelo CNN treinado
model_path = 'C:/Users/lucas/OneDrive/Área de Trabalho/Sistemas Inteligentes/ProjetoFinal_SI/modelo_flor25.keras'

# Tentativo do carregamento do modelo
try:
    model = tf.keras.models.load_model(model_path)
    print("Modelo carregado com sucesso!")
except OSError as e:
    print("Erro ao carregar o modelo:", e)

# Função para identificar a flor na imagem
def identify_flower(image_path):
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    prediction = model.predict(img_array)
    
    # Obtém os nomes das classes de flores
    class_names = sorted(os.listdir(data_dir))
    
    # Obtém as probabilidades para cada classe
    probabilities = prediction[0]
    
    # Cria um dicionário com as probabilidades associadas a cada classe
    probabilities_dict = {class_name: prob for class_name, prob in zip(class_names, probabilities)}
    
    # Obtém o nome da flor identificada com base na maior probabilidade
    predicted_class = np.argmax(prediction)
    flower_name = class_names[predicted_class]
    
    return flower_name, probabilities_dict

# Informações de cuidados para cada tipo de flor
care_information = {
    'Rosa': {
        'Rega': 'Regue a rosa regularmente, mantendo o solo úmido. ',
        'Luz Solar': 'Forneça luz solar direta, pelo menos 6 horas por dia.',
        'Adubação': 'Fertilize durante a estação de crescimento com adubo equilibrado.'
    },
    'Margarida': {
        'Solo': 'Prefere solo bem drenado, evite solo muito úmido.',
        'Luz Solar': 'Gosta de luz solar direta, mas pode tolerar sombra parcial.',
        'Rega': 'Regue quando o solo estiver seco ao toque.'
    },
    'Girassol': {
        'Solo': 'Prospere em solo fértil e bem drenado.',
        'Luz Solar': 'Necessita de luz solar plena, siga o sol.',
        'Rega': 'Regue regularmente, evite solo encharcado.'
    },
    'Astible': {
        'Solo': 'Prefere solo úmido e bem drenado.',
        'Luz Solar': 'Pode tolerar sombra parcial.',
        'Rega': 'Mantenha o solo úmido, evitando o encharcamento.'
    },
    'Calêndula': {
        'Solo': 'Prefere solo bem drenado.',
        'Luz Solar': 'Gosta de luz solar plena.',
        'Rega': 'Regue regularmente, permitindo que o solo seque entre as regas.'
    },
    'California Poppy': {
        'Solo': 'Adapta-se a solo pobre e arenoso.',
        'Luz Solar': 'Prefere luz solar plena.',
        'Rega': 'Tolerante à seca, regue moderadamente.'
    },
    'Campanula': {
        'Solo': 'Prefere solo úmido e bem drenado.',
        'Luz Solar': 'Pode tolerar sombra parcial.',
        'Rega': 'Mantenha o solo uniformemente úmido.'
    },
    'Coreopsis': {
        'Solo': 'Adapta-se a diferentes tipos de solo.',
        'Luz Solar': 'Prefere luz solar plena.',
        'Rega': 'Tolerante à seca, regue quando o solo estiver seco.'
    },
    'Cravo': {
        'Solo': 'Prefere solo bem drenado e rico.',
        'Luz Solar': 'Gosta de luz solar plena.',
        'Rega': 'Mantenha o solo uniformemente úmido.'
    },
    'Daffodil': {
        'Solo': 'Prefere solo bem drenado.',
        'Luz Solar': 'Gosta de luz solar plena ou parcial.',
        'Rega': 'Regue regularmente durante a estação de crescimento.'
    },
    'Dente de Leão': {
        'Solo': 'Adapta-se a diferentes tipos de solo.',
        'Luz Solar': 'Pode tolerar sombra parcial.',
        'Rega': 'Tolerante à seca, regue conforme necessário.'
    },
    'Íris': {
        'Solo': 'Prefere solo úmido e bem drenado.',
        'Luz Solar': 'Gosta de luz solar plena.',
        'Rega': 'Mantenha o solo uniformemente úmido.'
    },
    'Magnolia': {
        'Solo': 'Prefere solo bem drenado e ácido.',
        'Luz Solar': 'Pode tolerar sombra parcial.',
        'Rega': 'Regue regularmente, especialmente durante períodos secos.'
    },
    'Margarida Amarela': {
        'Solo': 'Prefere solo bem drenado.',
        'Luz Solar': 'Gosta de luz solar plena.',
        'Rega': 'Regue regularmente, permitindo que o solo seque entre as regas.'
    },
    'Tulipa': {
        'Solo': 'Prefere solo bem drenado.',
        'Luz Solar': 'Gosta de luz solar plena.',
        'Rega': 'Regue quando o solo estiver seco, especialmente durante o crescimento ativo.'
    },
    'Water Lily': {
        'Solo': 'Cresce em água ou solo encharcado.',
        'Luz Solar': 'Prefere luz solar plena.',
        'Rega': 'Mantenha a água ao redor da base.'
    }
}

# Função para processar a imagem e exibir as informações
def process_image():
    flower_name, probabilities_dict = identify_flower(image_path)
    
    label_result1.config(text=f"A flor na imagem é: ", font=('calibri', 14,'bold'), bg=root.cget('bg'))
    label_result.config(text=f"{flower_name}", font=('calibri', 16), bg=root.cget('bg'))

    care_info = care_information.get(flower_name, {})
    care_text = "\n".join([f"{aspect}: {info}" for aspect, info in care_info.items()])
    
    # Atualize o widget Label diretamente com o texto formatado
    label_care1.config(text=f"Cuidados: ", font=('calibri', 14, 'bold'), bg=root.cget('bg'))
    label_care.config(text=f"{care_text}", font=('calibri', 14), bg=root.cget('bg'))
    
    plot_probabilities_bar_chart(probabilities_dict)

    label_result1.pack()
    label_result.pack()
    label_care1.pack()
    label_care.pack()

# Função para escolher a imagem
def choose_image():
    global image_path
    initial_dir = 'C:/Users/lucas/OneDrive/Área de Trabalho/ProjetoFinal_SI/Imagem_teste'
    image_path = filedialog.askopenfilename(initialdir=initial_dir, title="Escolha a imagem")
    if image_path:
        img = Image.open(image_path)
        img = img.resize((250, 250))
        photo = ImageTk.PhotoImage(img)
        label_image.config(image=photo)
        label_image.image = photo
        label_result1.config(text="")
        label_result.config(text="")
        label_care1.config(text="")
        label_care.config(text="")

def plot_probabilities_bar_chart(probabilities_dict):
    plt.figure(figsize=(8, 6))
    classes = list(probabilities_dict.keys())
    probabilities = list(probabilities_dict.values())
    plt.bar(classes, probabilities)
    plt.xlabel('Classes de Flores')
    plt.ylabel('Probabilidades')
    plt.title('Probabilidades de Identificação de Flores')
    plt.xticks(rotation=45)  # Rotaciona os rótulos do eixo x para melhor legibilidade
    plt.tight_layout()
    plt.show()

# Função para centralizar a janela
def center_window(root, width, height):
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    
    root.geometry(f'{width}x{height}+{x}+{y}')

# Configuração da interface gráfica
root = tk.Tk()
root.title("Identificador de Flores")

# Largura e altura desejadas para a janela
window_width = 650 # Largura
window_height = 700 # Altura 

center_window(root, window_width, window_height)

# Estilos para os botões
style = ttk.Style()
style.configure('TButton', font=('calibri', 12, 'bold'), padding=10, width=20)

# Criação dos elementos da interface
label_image = tk.Label(root)
button_choose = ttk.Button(root, text="Escolher Imagem", command=choose_image)
button_identify = ttk.Button(root, text="Identificar", command=process_image)

label_result1 = tk.Label(root, text="")
label_result = tk.Label(root, text="")
label_care1 = tk.Label(root, text="")
label_care = tk.Label(root, text="")

empty_space = tk.Label(root, text="", pady=10, bg='#faf0e6')
empty_space2 = tk.Label(root, text="", pady=10, bg='#faf0e6')
empty_space3 = tk.Label(root, text="", pady=10, bg='#faf0e6')
empty_space4 = tk.Label(root, text="", pady=10, bg='#faf0e6')
empty_space5 = tk.Label(root, text="", pady=2, bg='#faf0e6')


# Posicionamento dos elementos na interface
empty_space.pack() # Separar a imagem do topo da tela
label_image.pack()

empty_space2.pack() # Separar o texto da imagem

label_result1.pack()
label_result.pack()

empty_space3.pack() # Separar 

label_care1.pack()
label_care.pack()

empty_space4.pack() # Separar a imagem dos botões

button_choose.pack()

empty_space5.pack() # Separar o botões

button_identify.pack()

# Configurar cor de fundo para a tela (root)
root.configure(bg='#faf0e6')  # Substitua com a cor desejada em formato hexadecimal

# Inicialização da interface
root.mainloop()