"""
Módulo de lógica de seguridad y análisis de riesgos
"""

import json
from datetime import datetime
import os


# Lista de objetos considerados sospechosos
# ADAPTADO a las clases del modelo de Igor: ['arma', 'gorro', 'mascara', 'persona']
SUSPICIOUS_OBJECTS = [
    'arma', 'weapon', 'knife', 'gun', 'mascara', 'gorro',
    'suspicious_package', 'unattended_baggage', 'restricted_item'
]


def check_security_risk(detections, risk_threshold=0.7):
    """
    Evalúa si las detecciones representan un riesgo de seguridad
    
    Args:
        detections: Lista de objetos detectados (desde detector.py de Igor)
                   Formato: [{'label': 'arma', 'confidence': 0.95}, ...]
        risk_threshold: Umbral de riesgo
    
    Returns:
        Dict con nivel de riesgo y objetos sospechosos
    """
    # Filtrar objetos sospechosos
    suspicious_objects = []
    
    for detection in detections:
        label = detection.get('label', '')
        confidence = detection.get('confidence', 0.0)
        
        if is_suspicious_object(label):
            suspicious_objects.append({
                'label': label,
                'confidence': confidence
            })
    
    # Calcular nivel de riesgo
    risk_level = calculate_risk_level(suspicious_objects)
    
    # Calcular score numérico
    risk_score = 0.0
    if suspicious_objects:
        total = sum(obj['confidence'] for obj in suspicious_objects)
        risk_score = total / len(suspicious_objects)
    
    return {
        'risk_level': risk_level,
        'suspicious_objects': suspicious_objects,
        'risk_score': risk_score,
        'alert_required': risk_score >= risk_threshold or len(suspicious_objects) > 0
    }


def calculate_risk_level(detected_objects):
    """
    Calcula el nivel de riesgo basado en los objetos detectados
    
    Args:
        detected_objects: Lista de objetos detectados
    
    Returns:
        Nivel de riesgo (bajo, medio, alto, crítico)
    """
    if not detected_objects:
        return 'bajo'
    
    # Si detecta arma, es crítico
    for obj in detected_objects:
        if obj.get('label', '').lower() in ['arma', 'weapon', 'gun', 'knife']:
            return 'crítico'
    
    # Si detecta máscara con alta confianza, es alto
    for obj in detected_objects:
        if obj.get('label', '').lower() == 'mascara' and obj.get('confidence', 0) > 0.7:
            return 'alto'
    
    # Si detecta gorro o máscara con baja confianza, es medio
    for obj in detected_objects:
        if obj.get('label', '').lower() in ['gorro', 'mascara']:
            return 'medio'
    
    # Por defecto
    return 'bajo'


def is_suspicious_object(object_label):
    """
    Determina si un objeto es considerado sospechoso
    
    Args:
        object_label: Etiqueta del objeto
    
    Returns:
        Boolean indicando si es sospechoso
    """
    object_label = object_label.lower().strip()
    
    # Verificar coincidencia exacta primero
    if object_label in SUSPICIOUS_OBJECTS:
        return True
    
    # Verificar si alguna palabra sospechosa está como palabra completa en el label
    for suspicious in SUSPICIOUS_OBJECTS:
        suspicious_lower = suspicious.lower()
        
        # Casos válidos:
        # - Coincidencia exacta: "arma" == "arma"
        # - Con guión bajo: "suspicious_package" contiene "suspicious"
        # - Con espacio: "big knife" contiene "knife"
        
        # Evitar falsos positivos como 'car' conteniendo 'ar' de 'arma'
        words_in_label = object_label.replace('_', ' ').split()
        
        if suspicious_lower in words_in_label:
            return True
    
    return False


def generate_alert(risk_level, suspicious_objects):
    """
    Genera una alerta de seguridad
    
    Args:
        risk_level: Nivel de riesgo detectado
        suspicious_objects: Lista de objetos sospechosos
    
    Returns:
        Mensaje de alerta formateado
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Mensaje según nivel
    if risk_level == 'crítico':
        header = "🚨🚨 ALERTA CRÍTICA"
    elif risk_level == 'alto':
        header = "🚨 ALERTA ALTA"
    elif risk_level == 'medio':
        header = "⚠️ ALERTA MEDIA"
    else:
        header = "ℹ️ MONITOREO"
    
    # Construir mensaje
    message = f"{header}\n"
    message += f"Timestamp: {timestamp}\n"
    message += f"Nivel de Riesgo: {risk_level.upper()}\n"
    message += f"Objetos detectados: {len(suspicious_objects)}\n"
    
    for obj in suspicious_objects:
        message += f"  - {obj['label']}: {obj['confidence']*100:.1f}%\n"
    
    return message


def log_security_event(event_data):
    """
    Registra un evento de seguridad
    
    Args:
        event_data: Datos del evento a registrar
    
    Returns:
        Confirmación del registro
    """
    # Agregar timestamp
    if 'timestamp' not in event_data:
        event_data['timestamp'] = datetime.now().isoformat()
    
    # Crear directorio si no existe
    log_file = 'logs/security_events.json'
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Leer eventos existentes
    events = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                events = json.load(f)
        except:
            events = []
    
    # Agregar nuevo evento
    events.append(event_data)
    
    # Guardar
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(events, f, indent=2, ensure_ascii=False)
        return {'success': True, 'message': 'Evento registrado'}
    except Exception as e:
        return {'success': False, 'message': str(e)}
