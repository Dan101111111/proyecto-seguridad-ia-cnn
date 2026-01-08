"""
Funciones de utilidad general
"""
import os
import json
from datetime import datetime


def save_image(image, output_path):
    """
    Guarda una imagen en disco
    
    Args:
        image: Imagen a guardar
        output_path: Ruta donde guardar la imagen
    
    Returns:
        Boolean indicando éxito
    """
    pass


def load_config(config_path='config.json'):
    """
    Carga la configuración desde un archivo JSON
    
    Args:
        config_path: Ruta al archivo de configuración
    
    Returns:
        Dict con la configuración
    """
    pass


def get_timestamp():
    """
    Obtiene la marca de tiempo actual
    
    Returns:
        String con timestamp formateado
    """
    pass


def create_output_directory(directory_path):
    """
    Crea un directorio si no existe
    
    Args:
        directory_path: Ruta del directorio a crear
    
    Returns:
        Ruta del directorio creado
    """
    pass


def format_detection_results(detections):
    """
    Formatea los resultados de detección para visualización
    
    Args:
        detections: Lista de detecciones
    
    Returns:
        String formateado con los resultados
    """
    pass


def validate_input(input_data):
    """
    Valida los datos de entrada
    
    Args:
        input_data: Datos a validar
    
    Returns:
        Boolean indicando si los datos son válidos
    """
    pass
