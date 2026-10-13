import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 1. Cargar el dataset de 50 restaurantes
df = pd.read_csv("restaurantes.csv")
df.columns = df.columns.str.strip()

# Definir las categorías de comida exactas
CATEGORIAS_COMIDA = ["Carne", "Tacos", "Pizza / Italiana", "Asiatica", "Mariscos"]

def codificar_comida(tipo_str):
    vector = [0.0] * len(CATEGORIAS_COMIDA)
    if tipo_str in CATEGORIAS_COMIDA:
        vector[CATEGORIAS_COMIDA.index(tipo_str)] = 1.0
    return vector

# 2. Generar datos sintéticos robustos de entrenamiento
X = []
y = []

num_restaurantes = len(df)

for idx, fila in df.iterrows():
    tipo = fila.get('tipo_comida', fila.get('tipo', 'Carne'))
    comida_vec = codificar_comida(tipo)
    
    # Valores base del restaurante normalizados
    amb_base = float(fila.get('ambiente', 2)) / 4.0
    pres_base = float(fila.get('presupuesto', 2)) / 4.0
    veg_base = float(fila.get('vegetariano', 0))
    distancia_norm = 0.1 

    vector_base = comida_vec + [amb_base, pres_base, veg_base, distancia_norm]
    
    # Ampliamos a 100 muestras por restaurante con ligera variación para mejorar la generalización
    for _ in range(100):
        variacion_amb = np.clip(amb_base + np.random.normal(0, 0.05), 0.0, 1.0)
        variacion_pres = np.clip(pres_base + np.random.normal(0, 0.05), 0.0, 1.0)
        
        vector_variante = comida_vec + [variacion_amb, variacion_pres, veg_base, distancia_norm]
        X.append(vector_variante)
        y.append(idx)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# 3. Construir la Red Neuronal (Perceptrón Multicapa optimizado)
model = Sequential([
    Dense(128, input_shape=(X.shape[1],), activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(32, activation='relu'),
    Dense(num_restaurantes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Entrenando modelo avanzado para {num_restaurantes} restaurantes...")
# Subimos a 100 épocas para afinar los pesos y mejorar el nivel de afinidad
model.fit(X, y, epochs=100, batch_size=16, verbose=1)

# 4. Guardar el modelo actualizado
model.save("modelo_restaurantes.keras")
print("¡Modelo reentrenado, optimizado y guardado exitosamente como 'modelo_restaurantes.keras'!")