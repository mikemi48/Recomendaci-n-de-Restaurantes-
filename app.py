import os
import traceback
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from geopy.distance import geodesic

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

# Cargar CSV limpiando espacios en blanco accidentales en los nombres de columnas
df = pd.read_csv("restaurantes.csv")
df.columns = df.columns.str.strip()

model = tf.keras.models.load_model("modelo_restaurantes.keras")

CATEGORIAS_COMIDA = ["Carne", "Tacos", "Pizza / Italiana", "Asiatica", "Mariscos"]

def codificar_comida(tipo_str):
    vector = [0.0] * len(CATEGORIAS_COMIDA)
    if tipo_str in CATEGORIAS_COMIDA:
        vector[CATEGORIAS_COMIDA.index(tipo_str)] = 1.0
    return vector

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recomendar', methods=['POST'])
def recomendar():
    try:
        datos = request.get_json(force=True)

        comida_vec = codificar_comida(datos.get('tipo_comida', 'Carne'))
        ambiente_norm = float(datos.get('ambiente', 1)) / 4.0
        presupuesto_norm = float(datos.get('presupuesto', 1)) / 4.0
        vegetariano_val = float(datos.get('vegetariano', 0))

        # 1. Coordenadas reales del usuario enviadas por el navegador
        lat_usr = float(datos.get('latitud', 25.6667))
        lon_usr = float(datos.get('longitud', -100.3167))

        # 2. Inferencia de afinidad con la red neuronal
        distancia_ref = 0.1  # Referencia de cercanía
        vector = comida_vec + [ambiente_norm, presupuesto_norm, vegetariano_val, distancia_ref]
        vector_entrada = np.array([vector], dtype=np.float32)

        predicciones = model.predict(vector_entrada, verbose=0)[0]

        # 3. Detectar nombres de columnas del CSV
        cols_lower = {col.lower(): col for col in df.columns}
        col_rest = cols_lower.get('restaurante', cols_lower.get('nombre', df.columns[1]))
        col_zona = cols_lower.get('zona', df.columns[2] if len(df.columns) > 2 else None)
        col_tipo = cols_lower.get('tipo_comida', cols_lower.get('tipo', None))
        col_lat = cols_lower.get('latitud', df.columns[-2])
        col_lon = cols_lower.get('longitud', df.columns[-1])

        # 4. Ponderar afinidad de la red con la cercanía en km
        mejor_score = -1.0
        mejor_idx = 0
        mejor_distancia = 0.0

        for i, fila in df.iterrows():
            prob_red = predicciones[i]
            d_km = geodesic((lat_usr, lon_usr), (float(fila[col_lat]), float(fila[col_lon]))).km
            
            # Factor de cercanía (1.0 si está a 0 km, disminuye con la distancia)
            factor_cercania = max(0.05, 1.0 - (d_km / 25.0))
            
            # Puntuación final: 60% peso red neuronal + 40% proximidad física
            score_final = (prob_red * 0.6) + (factor_cercania * 0.4)

            if score_final > mejor_score:
                mejor_score = score_final
                mejor_idx = i
                mejor_distancia = d_km

        restaurante = df.iloc[mejor_idx]
        afinidad_pct = float(predicciones[mejor_idx]) * 100.0

        return jsonify({
            "nombre": str(restaurante[col_rest]),
            "zona": str(restaurante[col_zona]) if col_zona else "Monterrey",
            "tipo_comida": str(restaurante[col_tipo]) if col_tipo else str(datos.get('tipo_comida')),
            "distancia": f"{mejor_distancia:.2f} km",
            "afinidad": f"{afinidad_pct:.1f}%"
        })

    except Exception as e:
        print("\n--- ERROR EN /recomendar ---")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)