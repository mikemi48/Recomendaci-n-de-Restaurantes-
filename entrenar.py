import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Cargar el archivo CSV de restaurantes
df = pd.read_csv("restaurantes.csv")
num_restaurantes = len(df)
print(f"Total de restaurantes cargados: {num_restaurantes}")

# Mapeo de categorías de comida a columnas One-Hot fijas
CATEGORIAS_COMIDA = ["Carne", "Tacos", "Pizza / Italiana", "Asiatica", "Mariscos"]

def codificar_comida(tipo_str):
    vector = [0] * len(CATEGORIAS_COMIDA)
    if tipo_str in CATEGORIAS_COMIDA:
        vector[CATEGORIAS_COMIDA.index(tipo_str)] = 1
    return vector

# 2. Generar datos sintéticos de entrenamiento
# Simulamos consultas de usuarios con preferencias y les asignamos el restaurante ideal
np.random.seed(42)
N_SAMPLES = 2500

X_train = []
y_train = []

for _ in range(N_SAMPLES):
    # Elegir aleatoriamente un restaurante objetivo del dataset
    idx_target = np.random.randint(0, num_restaurantes)
    rest_info = df.iloc[idx_target]
    
    # 5 entradas de comida (One-Hot)
    comida_vec = codificar_comida(rest_info['tipo_comida'])
    
    # Entradas normalizadas (escala de 0.0 a 1.0)
    ambiente_norm = float(rest_info['ambiente']) / 4.0
    presupuesto_norm = float(rest_info['presupuesto']) / 4.0
    vegetariano_val = float(rest_info['vegetariano'])
    
    # Simulamos que el usuario está a una distancia cercana a este restaurante (ej. entre 1 y 10 km)
    distancia_simulada_km = np.random.uniform(1.0, 10.0)
    distancia_norm = min(distancia_simulada_km / 25.0, 1.0)
    
    # Vector de entrada completo: 9 variables
    vector_entrada = comida_vec + [ambiente_norm, presupuesto_norm, vegetariano_val, distancia_norm]
    
    X_train.append(vector_entrada)
    y_train.append(idx_target)

X_train = np.array(X_train)
y_train = np.array(y_train)

# 3. Construir la arquitectura de la Red Neuronal (MLP)
# Entrada: 9 variables
# Capas ocultas: 64 neuronas -> Dropout -> 32 neuronas
# Capa de salida: N neuronas (una por cada restaurante en tu CSV)
model = models.Sequential([
    layers.Input(shape=(9,)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_restaurantes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--- Iniciando entrenamiento de la red neuronal ---")
history = model.fit(
    X_train, 
    y_train, 
    epochs=35, 
    batch_size=32, 
    verbose=1
)

# 4. Guardar el modelo entrenado en un archivo
model.save("modelo_restaurantes.keras")
print("\n¡Entrenamiento completado exitosamente!")
print("Se ha generado el archivo 'modelo_restaurantes.keras' en tu carpeta.")