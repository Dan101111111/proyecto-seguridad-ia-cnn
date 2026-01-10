"""
Módulo de funciones auxiliares para el sistema de detección de seguridad.
Utilidades generales para el proyecto.
Autor: D'Alessandro
"""

import json
import yaml
import logging
import os
import cv2
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> logging.Logger:
    """
    Configura el sistema de logging para el proyecto.
    
    Args:
        log_dir: Directorio donde guardar los logs.
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        
    Returns:
        Logger configurado.
    """
    # Crear directorio de logs si no existe
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Configurar nombre del archivo de log con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"detection_system_{timestamp}.log"
    
    # Configurar formato
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configurar logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Sistema de logging iniciado. Log guardado en: {log_file}")
    
    return logger


def save_image(image, output_path):
    """
    Guarda una imagen en disco
    
    Args:
        image: Imagen a guardar
        output_path: Ruta donde guardar la imagen
    
    Returns:
        Boolean indicando éxito
    """
    try:
        # Crear directorio si no existe
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convertir de RGB a BGR si es necesario
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        # Guardar imagen
        success = cv2.imwrite(str(output_path), image_bgr)
        return success
    except Exception as e:
        print(f"Error al guardar imagen: {str(e)}")
        return False


def load_config(config_path='config.json'):
    """
    Carga la configuración desde un archivo JSON o YAML
    
    Args:
        config_path: Ruta al archivo de configuración
    
    Returns:
        Dict con la configuración
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        print(f"Advertencia: Archivo de configuración no encontrado: {config_path}")
        return {}
    
    try:
        # Cargar según extensión
        if config_file.suffix in ['.yaml', '.yml']:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        elif config_file.suffix == '.json':
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            print(f"Formato de configuración no soportado: {config_file.suffix}")
            return {}
        
        return config
    except Exception as e:
        print(f"Error al cargar configuración: {str(e)}")
        return {}


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Guarda un diccionario de configuración en un archivo.
    
    Args:
        config: Diccionario con la configuración.
        config_path: Ruta donde guardar el archivo.
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Guardar según extensión
    if config_file.suffix in ['.yaml', '.yml']:
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif config_file.suffix == '.json':
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
    else:
        raise ValueError(f"Formato de configuración no soportado: {config_file.suffix}")


def get_timestamp():
    """
    Obtiene la marca de tiempo actual
    
    Returns:
        String con timestamp formateado
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_output_directory(directory_path):
    """
    Crea un directorio si no existe
    
    Args:
        directory_path: Ruta del directorio a crear
    
    Returns:
        Ruta del directorio creado
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def format_detection_results(detections):
    """
    Formatea los resultados de detección para visualización
    
    Args:
        detections: Lista de detecciones
    
    Returns:
        String formateado con los resultados
    """
    if not detections:
        return "No se detectaron objetos"
    
    result = "\n=== RESULTADOS DE DETECCIÓN ===\n"
    for i, detection in enumerate(detections, 1):
        result += f"\nDetección {i}:\n"
        result += f"  - Clase: {detection.get('class', 'N/A')}\n"
        result += f"  - Confianza: {detection.get('confidence', 0):.2%}\n"
        if 'bbox' in detection:
            bbox = detection['bbox']
            result += f"  - Ubicación: ({bbox[0]}, {bbox[1]}) - ({bbox[2]}, {bbox[3]})\n"
    
    return result


def validate_input(input_data):
    """
    Valida los datos de entrada
    
    Args:
        input_data: Datos a validar
    
    Returns:
        Boolean indicando si los datos son válidos
    """
    if input_data is None:
        return False
    
    # Si es una imagen (numpy array)
    if isinstance(input_data, np.ndarray):
        # Verificar que tenga dimensiones válidas
        if len(input_data.shape) < 2:
            return False
        # Verificar que no esté vacía
        if input_data.size == 0:
            return False
        return True
    
    # Si es una ruta de archivo
    if isinstance(input_data, (str, Path)):
        path = Path(input_data)
        return path.exists() and path.is_file()
    
    return True


def create_directory_structure(base_dir: str) -> None:
    """
    Crea la estructura de directorios necesaria para el proyecto.
    
    Args:
        base_dir: Directorio base del proyecto.
    """
    directories = [
        "data/raw",
        "data/processed",
        "models/checkpoints",
        "models/final",
        "logs",
        "results/images",
        "results/metrics"
    ]
    
    base_path = Path(base_dir)
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Directorio creado/verificado: {dir_path}")


def get_class_names(data_dir: str) -> List[str]:
    """
    Obtiene los nombres de las clases del dataset.
    
    Args:
        data_dir: Directorio raíz del dataset.
        
    Returns:
        Lista con los nombres de las clases.
    """
    data_path = Path(data_dir)
    class_names = [d.name for d in data_path.iterdir() if d.is_dir()]
    class_names.sort()
    return class_names


def count_samples_per_class(data_dir: str) -> Dict[str, int]:
    """
    Cuenta el número de muestras por clase en el dataset.
    
    Args:
        data_dir: Directorio raíz del dataset.
        
    Returns:
        Diccionario con el conteo por clase.
    """
    data_path = Path(data_dir)
    counts = {}
    
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            # Contar archivos de imagen
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            count = sum(len(list(class_dir.glob(ext))) for ext in image_extensions)
            counts[class_dir.name] = count
    
    return counts


def plot_class_distribution(class_counts: Dict[str, int], save_path: Optional[str] = None) -> None:
    """
    Visualiza la distribución de clases en el dataset.
    
    Args:
        class_counts: Diccionario con el conteo por clase.
        save_path: Ruta donde guardar la imagen (opcional).
    """
    plt.figure(figsize=(10, 6))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.bar(classes, counts, color='skyblue', edgecolor='navy')
    plt.xlabel('Clase')
    plt.ylabel('Número de muestras')
    plt.title('Distribución de Clases en el Dataset')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    class_names: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    Visualiza la matriz de confusión.
    
    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Etiquetas predichas.
        class_names: Nombres de las clases.
        save_path: Ruta donde guardar la imagen (opcional).
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta Real')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matriz de confusión guardada en: {save_path}")
    
    plt.show()


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
) -> str:
    """
    Genera un reporte de clasificación detallado.
    
    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Etiquetas predichas.
        class_names: Nombres de las clases.
        save_path: Ruta donde guardar el reporte (opcional).
        
    Returns:
        String con el reporte de clasificación.
    """
    from sklearn.metrics import classification_report
    
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Reporte guardado en: {save_path}")
    
    return report


def visualize_predictions(
    images: List[np.ndarray],
    true_labels: List[str],
    pred_labels: List[str],
    confidences: List[float],
    save_path: Optional[str] = None,
    num_images: int = 8
) -> None:
    """
    Visualiza predicciones del modelo.
    
    Args:
        images: Lista de imágenes.
        true_labels: Etiquetas verdaderas.
        pred_labels: Etiquetas predichas.
        confidences: Confianzas de las predicciones.
        save_path: Ruta donde guardar la imagen (opcional).
        num_images: Número de imágenes a mostrar.
    """
    num_images = min(num_images, len(images))
    cols = 4
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten() if num_images > 1 else [axes]
    
    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i])
        
        # Color verde si es correcta, rojo si es incorrecta
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        
        title = f"Real: {true_labels[i]}\nPred: {pred_labels[i]}\nConf: {confidences[i]:.2%}"
        ax.set_title(title, color=color, fontsize=10)
        ax.axis('off')
    
    # Ocultar ejes vacíos
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualización guardada en: {save_path}")
    
    plt.show()


def calculate_model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas básicas del modelo.
    
    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Etiquetas predichas.
        
    Returns:
        Diccionario con las métricas.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    
    return metrics


def format_time(seconds: float) -> str:
    """
    Formatea tiempo en segundos a un formato legible.
    
    Args:
        seconds: Tiempo en segundos.
        
    Returns:
        String con el tiempo formateado.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_device() -> str:
    """
    Obtiene el dispositivo disponible para entrenamiento (GPU o CPU).
    
    Returns:
        String con el nombre del dispositivo ('cuda' o 'cpu').
    """
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Usando dispositivo: {device}")
        if device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        return device
    except ImportError:
        print("PyTorch no está instalado. Usando CPU por defecto.")
        return 'cpu'


def print_system_info() -> None:
    """Imprime información del sistema y librerías instaladas."""
    import sys
    
    print("=" * 50)
    print("INFORMACIÓN DEL SISTEMA")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"OpenCV version: {cv2.__version__}")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Dispositivos CUDA: {torch.cuda.device_count()}")
    except ImportError:
        print("PyTorch no está instalado")
    
    print("=" * 50)
