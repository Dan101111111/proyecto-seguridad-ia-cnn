"""
Módulo de detección de objetos usando CNN
"""
import tensorflow as tf
import numpy as np
import cv2
# Importamos la función de D'Alessandro para que todo sea coherente
from src.preprocessing import preprocess_frame 

def load_model(model_path='models/modelo_seguridad_v4.keras'):
    """ Carga el modelo entrenado """
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

def detect_objects(frame, model, threshold=0.5, enable_region_analysis=True):
    """
    Detecta múltiples objetos en la imagen.
    MODO SEGURIDAD: Usa threshold adaptativo + análisis de regiones para objetos peligrosos.
    """
    if model is None:
        return []

    # 1. Predicción global
    input_frame = preprocess_frame(frame)
    input_frame_batch = np.expand_dims(input_frame, axis=0)
    predictions = model.predict(input_frame_batch, verbose=0)
    
    # 2. Formatear resultados con THRESHOLDS ADAPTATIVOS
    clases = ['arma', 'gorro', 'mascara', 'persona']
    
    # Thresholds optimizados para detección de personas con armas
    thresholds = {
        'arma': 0.15,       # 15% - Balance para detección global
        'gorro': 0.25,      # 25%
        'mascara': 0.25,    # 25%
        'persona': 0.50     # 50%
    }
    
    results = []
    max_prob_idx = np.argmax(predictions[0])
    
    # Detectar con thresholds adaptativos
    for idx, probabilidad in enumerate(predictions[0]):
        clase = clases[idx]
        umbral = thresholds.get(clase, threshold)
        
        if probabilidad > umbral:
            results.append({
                'label': clase,
                'confidence': float(probabilidad),
                'is_primary': bool(idx == max_prob_idx),  # Convertir a bool nativo de Python
                'source': 'global'
            })
    
    # 3. ANÁLISIS REGIONAL INTELIGENTE (para detectar armas en manos)
    regional_detections = []
    regional_arma_probs = []
    
    if enable_region_analysis:
        regional_detections = analyze_image_regions(frame, model, thresholds)
        regional_arma_probs = [det['confidence'] for det in regional_detections if det['label'] == 'arma']
        
        # Validación inteligente: distinguir arma real vs gesto de mano
        if regional_arma_probs:
            arma_global = float(predictions[0][0])
            persona_global = float(predictions[0][3])
            
            # Criterios para validar arma:
            # 1. Si el modelo global NO ve arma (<5%), pero regional sí → DESCONFIAR
            # 2. Si múltiples regiones detectan arma → MÁS confiable
            # 3. Si una sola región con alta confianza pero global bajo → Probablemente falso positivo
            
            num_regiones_con_arma = len(regional_arma_probs)
            max_regional = max(regional_arma_probs)
            
            # FILTRO PRINCIPAL: Si global es MUY bajo (<5%), requerir MÚLTIPLES regiones
            if arma_global < 0.05:
                # Solo aceptar si 2+ regiones detectan arma con confianza razonable
                if num_regiones_con_arma < 2:
                    # Una sola región pero global bajo → RECHAZAR (probable gesto de mano)
                    regional_detections = [d for d in regional_detections if d['label'] != 'arma']
                elif num_regiones_con_arma == 2 and max_regional < 0.50:
                    # Dos regiones pero ninguna con alta confianza → DESCONFIAR
                    regional_detections = [d for d in regional_detections if d['label'] != 'arma']
            
            # FILTRO SECUNDARIO: Global moderado (5-10%) pero solo 1 región con confianza baja
            elif arma_global < 0.10 and num_regiones_con_arma == 1 and max_regional < 0.40:
                regional_detections = [d for d in regional_detections if d['label'] != 'arma']
            
            # CASO VÁLIDO: 
            # - Global alto (>=10%) → Confiar en detección
            # - Múltiples regiones con alta confianza → Confiar
            # - Global bajo pero 2+ regiones con confianza alta → Confiar
            else:
                # Mantener las detecciones regionales válidas
                pass
        
        # Agregar detecciones regionales validadas
        for det in regional_detections:
            if not any(r['label'] == det['label'] for r in results):
                results.append(det)
            elif det['confidence'] > max((r['confidence'] for r in results if r['label'] == det['label']), default=0):
                results = [r for r in results if r['label'] != det['label']]
                results.append(det)
    
    # Ordenar por confianza descendente
    results.sort(key=lambda x: x['confidence'], reverse=True)
    
    return results


def validate_weapon_detection(arma_prob, persona_prob, max_regional_arma):
    """
    Valida si la detección de arma es real o un falso positivo.
    
    Sistema BALANCEADO que detecta armas reales sin generar falsos positivos.
    
    Estrategia:
    - Acepta evidencia regional MUY FUERTE (>20%)
    - Acepta evidencia global ALTA (>=8%)
    - Filtra detecciones débiles cuando persona domina
    
    Args:
        arma_prob: Probabilidad global de arma
        persona_prob: Probabilidad global de persona
        max_regional_arma: Máxima probabilidad regional de arma
    
    Returns:
        dict con 'is_valid', 'confidence_level', 'reason'
    """
    # REGLA 1: Evidencia regional MUY FUERTE (>=20%) → VÁLIDO
    if max_regional_arma >= 0.20:  # 20%
        return {
            'is_valid': True,
            'confidence_level': 'alta' if max_regional_arma >= 0.30 else 'media',
            'reason': f'Evidencia regional muy fuerte ({max_regional_arma:.1%})'
        }
    
    # REGLA 2: Probabilidad global ALTA (>=8%) → VÁLIDO
    if arma_prob >= 0.08:
        return {
            'is_valid': True,
            'confidence_level': 'alta' if arma_prob >= 0.20 else 'media',
            'reason': f'Detección global confirmada ({arma_prob:.1%})'
        }
    
    # REGLA 3: Persona domina (>99%) con evidencia débil → RECHAZAR
    if persona_prob > 0.99:
        if arma_prob < 0.01 and max_regional_arma < 0.20:
            return {
                'is_valid': False,
                'confidence_level': 'muy_baja',
                'reason': f'Persona domina ({persona_prob:.1%}), evidencia de arma insuficiente'
            }
    
    # REGLA 4: Evidencia muy débil → RECHAZAR
    if arma_prob < 0.001 and max_regional_arma < 0.15:
        return {
            'is_valid': False,
            'confidence_level': 'muy_baja',
            'reason': f'Evidencia muy débil (global:{arma_prob:.2%}, regional:{max_regional_arma:.1%})'
        }
    
    # REGLA 5: Evidencia combinada moderada (global >= 2% Y regional >= 15%)
    if arma_prob >= 0.02 and max_regional_arma >= 0.15:
        return {
            'is_valid': True,
            'confidence_level': 'media',
            'reason': f'Evidencia combinada (global:{arma_prob:.2%}, regional:{max_regional_arma:.1%})'
        }
    
    # Por defecto: rechazar
    return {
        'is_valid': False,
        'confidence_level': 'baja',
        'reason': f'Evidencia insuficiente (global:{arma_prob:.2%}, regional:{max_regional_arma:.1%})'
    }


def analyze_image_regions(frame, model, thresholds):
    """
    Analiza regiones específicas de la imagen para detectar objetos pequeños (ej: armas)
    """
    detections = []
    height, width = frame.shape[:2]
    
    # Dividir en regiones: superior-derecha, superior-izquierda, centro
    regions = [
        ('upper_right', frame[0:height//2, width//2:width]),   # Manos extendidas
        ('upper_left', frame[0:height//2, 0:width//2]),
        ('center', frame[height//4:3*height//4, width//4:3*width//4])
    ]
    
    for region_name, region_img in regions:
        if region_img.size == 0:
            continue
            
        # Predecir en región
        region_processed = preprocess_frame(region_img)
        region_batch = np.expand_dims(region_processed, axis=0)
        region_pred = model.predict(region_batch, verbose=0)
        
        # Buscar armas en regiones con threshold moderado
        # La validación posterior filtrará falsos positivos
        arma_prob = float(region_pred[0][0])
        regional_threshold = 0.25  # 25% - Moderado, validación filtrará falsos positivos
        
        if arma_prob > regional_threshold:
            detections.append({
                'label': 'arma',
                'confidence': arma_prob,
                'is_primary': False,
                'source': f'region_{region_name}',
                'warning': f'Detectada en región {region_name}'
            })
    
    return detections


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
