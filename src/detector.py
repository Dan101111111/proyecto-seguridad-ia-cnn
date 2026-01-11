"""
Módulo de detección de objetos usando CNN
"""
import tensorflow as tf
import numpy as np
import cv2
# Importamos la función de D'Alessandro para que todo sea coherente
from src.preprocessing import preprocess_frame 

def load_model(model_path='models/modelo_seguridad_v1.h5'):
    """ Carga el modelo entrenado por Igor """
    try:
        model = tf.keras.models.load_model(model_path)
        print("Modelo cargado correctamente.")
        return model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

def detect_objects(frame, model, threshold=0.6):
    """
    Recibe un frame de la cámara, lo procesa con la lógica de D'Alessandro
    y predice usando el modelo de Igor.
    """
    if model is None:
        return []

    # 1. Preprocesamiento (Estándar D'Alessandro 224x224)
    # Suponiendo que su función devuelve la imagen normalizada
    input_frame = preprocess_frame(frame)
    input_frame = np.expand_dims(input_frame, axis=0) # Ajustar para el modelo

    # 2. Predicción
    predictions = model.predict(input_frame, verbose=0)
    
    # 3. Formatear resultados
    # clases: 0: arma, 1: gorro, 2: mascara, 3: persona (el orden de las carpetas)
    clases = ['arma', 'gorro', 'mascara', 'persona']
    idx = np.argmax(predictions[0])
    probabilidad = predictions[0][idx]

    results = []
    if probabilidad > threshold:
        results.append({
            'label': clases[idx],
            'confidence': probabilidad
        })
    
    return results
