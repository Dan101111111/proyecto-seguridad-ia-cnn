"""
Paquete principal del sistema de detección de seguridad con CNN.
"""

__version__ = "1.0.0"
__author__ = "Equipo de Seguridad IA"

from src.detector import load_model, detect_objects
from src.preprocessing import preprocess_frame
from src.logic import check_security_risk, calculate_risk_level
from src.utils import save_image, get_timestamp, load_config

__all__ = [
    'load_model',
    'detect_objects',
    'preprocess_frame',
    'check_security_risk',
    'calculate_risk_level',
    'save_image',
    'get_timestamp',
    'load_config'
]
