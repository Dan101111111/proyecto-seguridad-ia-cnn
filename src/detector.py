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
        # Intentar cargar con compile=False para evitar problemas de compatibilidad
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Modelo cargado correctamente.")
        
        # Recompilar el modelo con configuración básica
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        print("\n💡 SUGERENCIAS:")
        print("1. Verifica que el archivo del modelo sea compatible con TensorFlow 2.x")
        print("2. Regenera el modelo usando el mismo TensorFlow que tienes instalado")
        print("3. Contacta a Igor para que verifique la arquitectura del modelo")
        return None

def detect_objects(frame, model, threshold=0.5):
    """
    Detecta múltiples objetos en la imagen.
    MODIFICADO: Ahora detecta TODAS las clases con confianza > threshold
    para capturar escenarios como "persona + arma" en seguridad.
    """
    if model is None:
        return []

    # 1. Preprocesamiento (Estándar D'Alessandro 224x224)
    input_frame = preprocess_frame(frame)
    input_frame = np.expand_dims(input_frame, axis=0)

    # 2. Predicción
    predictions = model.predict(input_frame, verbose=0)
    
    # 3. Formatear resultados - MULTI-LABEL
    # clases: 0: arma, 1: gorro, 2: mascara, 3: persona
    clases = ['arma', 'gorro', 'mascara', 'persona']
    
    results = []
    
    # Detectar TODAS las clases con confianza alta (no solo la máxima)
    for idx, probabilidad in enumerate(predictions[0]):
        if probabilidad > threshold:
            results.append({
                'label': clases[idx],
                'confidence': float(probabilidad)
            })
    
    # Ordenar por confianza descendente
    results.sort(key=lambda x: x['confidence'], reverse=True)
    
    return results


def draw_detections(frame, detections, color=(0, 255, 0), thickness=2):
    """
    Dibuja las detecciones sobre el frame (para visualización en Streamlit).
    
    Args:
        frame: Imagen original
        detections: Lista de detecciones [{'label': 'arma', 'confidence': 0.95}, ...]
        color: Color del texto y borde (RGB)
        thickness: Grosor de la línea
    
    Returns:
        Frame con las detecciones dibujadas
    """
    import cv2
    
    # Crear una copia para no modificar el original
    output_frame = frame.copy()
    
    if not detections:
        return output_frame
    
    # Posición para mostrar las etiquetas
    y_offset = 30
    
    for detection in detections:
        label = detection.get('label', 'Desconocido')
        confidence = detection.get('confidence', 0.0)
        
        # Crear texto
        text = f"{label}: {confidence:.2%}"
        
        # Obtener tamaño del texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Dibujar fondo del texto (rectángulo)
        cv2.rectangle(
            output_frame,
            (10, y_offset - text_height - 10),
            (10 + text_width + 10, y_offset + baseline),
            (0, 0, 0),
            -1  # Relleno
        )
        
        # Dibujar texto
        cv2.putText(
            output_frame,
            text,
            (15, y_offset - 5),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )
        
        # Ajustar posición para siguiente etiqueta
        y_offset += text_height + baseline + 20
    
    return output_frame


def get_detection_summary(detections):
    """
    Genera un resumen de las detecciones.
    
    Args:
        detections: Lista de detecciones
    
    Returns:
        Dict con resumen de detecciones
    """
    if not detections:
        return {
            'total': 0,
            'labels': [],
            'max_confidence': 0.0,
            'avg_confidence': 0.0
        }
    
    labels = [d.get('label', 'Desconocido') for d in detections]
    confidences = [d.get('confidence', 0.0) for d in detections]
    
    return {
        'total': len(detections),
        'labels': labels,
        'max_confidence': max(confidences) if confidences else 0.0,
        'avg_confidence': sum(confidences) / len(confidences) if confidences else 0.0
    }
