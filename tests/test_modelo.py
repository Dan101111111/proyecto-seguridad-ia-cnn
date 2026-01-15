import cv2
import numpy as np
import tensorflow as tf
from src.preprocessing import preprocess_frame 

# 1. Configuración
MODEL_PATH = 'models/modelo_seguridad_v1.h5'
# Pon aquí el nombre de una imagen que tengas para probar
IMAGE_TEST = 'data/arma/test_1.jpg' 

def test_inference():
    print("--- TEST DE MODELO CNN ---")
    
    # Cargar modelo
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("[OK] Modelo cargado.")
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo: {e}")
        return

    # Cargar y preprocesar imagen
    image = cv2.imread(IMAGE_TEST)
    if image is None:
        print(f"[ERROR] No se encontró la imagen en {IMAGE_TEST}")
        return

    # Usamos la lógica de D'Alessandro que ya está integrada
    processed = preprocess_frame(image)
    input_data = np.expand_dims(processed, axis=0)

    # Predicción
    preds = model.predict(input_data, verbose=0)
    clases = ['arma', 'gorro', 'mascara', 'persona']
    idx = np.argmax(preds[0])
    confianza = preds[0][idx]

    print("-" * 30)
    print(f"RESULTADO DEL TEST:")
    print(f"Objeto detectado: {clases[idx].upper()}")
    print(f"Confianza: {confianza:.2%}")
    print("-" * 30)

if __name__ == "__main__":
    test_inference()
