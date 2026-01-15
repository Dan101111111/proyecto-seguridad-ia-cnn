"""Módulo de preprocesamiento de datos para el sistema de detección de seguridad.
Limpieza de datos y transformaciones de imágenes.
Autor: D'Alessandro
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ImagePreprocessor:
    """Clase para el preprocesamiento de imágenes del dataset."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Inicializa el preprocesador de imágenes.
        
        Args:
            target_size: Tamaño objetivo (ancho, alto) para redimensionar las imágenes.
        """
        self.target_size = target_size
        
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Carga una imagen desde el disco.
        
        Args:
            image_path: Ruta a la imagen.
            
        Returns:
            Imagen en formato numpy array (RGB) o None si hay error.
        """
        try:
            # Leer imagen
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Error: No se pudo cargar la imagen {image_path}")
                return None
            
            # Convertir de BGR a RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            print(f"Error al cargar imagen {image_path}: {str(e)}")
            return None
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Redimensiona una imagen al tamaño objetivo.
        
        Args:
            image: Imagen en formato numpy array.
            
        Returns:
            Imagen redimensionada.
        """
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normaliza los valores de píxeles al rango [0, 1].
        
        Args:
            image: Imagen en formato numpy array.
            
        Returns:
            Imagen normalizada.
        """
        return image.astype(np.float32) / 255.0
    
    def remove_noise(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Elimina ruido de la imagen usando filtro Gaussiano.
        
        Args:
            image: Imagen en formato numpy array.
            kernel_size: Tamaño del kernel para el filtro.
            
        Returns:
            Imagen sin ruido.
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def adjust_brightness_contrast(
        self, 
        image: np.ndarray, 
        brightness: float = 0, 
        contrast: float = 1.0
    ) -> np.ndarray:
        """
        Ajusta el brillo y contraste de la imagen.
        
        Args:
            image: Imagen en formato numpy array.
            brightness: Factor de ajuste de brillo (-100 a 100).
            contrast: Factor de ajuste de contraste (0.5 a 2.0).
            
        Returns:
            Imagen ajustada.
        """
        adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return adjusted
    
    def preprocess_image(
        self, 
        image_path: str, 
        normalize: bool = True,
        remove_noise_flag: bool = False
    ) -> Optional[np.ndarray]:
        """
        Pipeline completo de preprocesamiento de una imagen.
        
        Args:
            image_path: Ruta a la imagen.
            normalize: Si se debe normalizar la imagen.
            remove_noise_flag: Si se debe eliminar el ruido.
            
        Returns:
            Imagen preprocesada o None si hay error.
        """
        # Cargar imagen
        image = self.load_image(image_path)
        if image is None:
            return None
        
        # Redimensionar
        image = self.resize_image(image)
        
        # Eliminar ruido si se solicita
        if remove_noise_flag:
            image = self.remove_noise(image)
        
        # Normalizar si se solicita
        if normalize:
            image = self.normalize_image(image)
        
        return image


class DataAugmentor:
    """Clase para aplicar técnicas de data augmentation."""
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        """
        Inicializa el augmentor de datos.
        
        Args:
            image_size: Tamaño de las imágenes (ancho, alto).
        """
        self.image_size = image_size
        
    def get_train_transforms(self) -> A.Compose:
        """
        Obtiene las transformaciones para el conjunto de entrenamiento.
        
        Returns:
            Composición de transformaciones de Albumentations.
        """
        return A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.RandomGamma(p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def get_validation_transforms(self) -> A.Compose:
        """
        Obtiene las transformaciones para el conjunto de validación.
        
        Returns:
            Composición de transformaciones de Albumentations.
        """
        return A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def apply_transforms(self, image: np.ndarray, transforms: A.Compose) -> np.ndarray:
        """
        Aplica transformaciones a una imagen.
        
        Args:
            image: Imagen en formato numpy array.
            transforms: Composición de transformaciones.
            
        Returns:
            Imagen transformada.
        """
        transformed = transforms(image=image)
        return transformed['image']


# Funciones independientes para compatibilidad con el resto del proyecto

def preprocess_frame(frame, target_size=(224, 224)):
    """
    Preprocesa un frame para alimentar al modelo CNN
    
    Args:
        frame: Frame/imagen a preprocesar
        target_size: Tamaño objetivo para redimensionar
    
    Returns:
        Frame preprocesado
    """
    if frame is None:
        return None
    
    # Redimensionar
    resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalizar a [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    return normalized


def resize_frame(frame, width, height):
    """
    Redimensiona un frame a las dimensiones especificadas
    
    Args:
        frame: Frame a redimensionar
        width: Ancho objetivo
        height: Alto objetivo
    
    Returns:
        Frame redimensionado
    """
    if frame is None:
        return None
    
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def normalize_frame(frame):
    """
    Normaliza los valores de píxeles del frame
    
    Args:
        frame: Frame a normalizar
    
    Returns:
        Frame normalizado
    """
    if frame is None:
        return None
    
    # Normalizar a rango [0, 1]
    normalized = frame.astype(np.float32) / 255.0
    
    return normalized


def apply_filters(frame, filter_type='blur'):
    """
    Aplica filtros de procesamiento de imagen
    
    Args:
        frame: Frame a procesar
        filter_type: Tipo de filtro ('blur', 'sharpen', 'edge')
    
    Returns:
        Frame con filtro aplicado
    """
    if frame is None:
        return None
    
    if filter_type == 'blur':
        # Filtro Gaussiano para suavizar
        return cv2.GaussianBlur(frame, (5, 5), 0)
    
    elif filter_type == 'sharpen':
        # Kernel para afilar
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(frame, -1, kernel)
    
    elif filter_type == 'edge':
        # Detección de bordes con Canny
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
        edges = cv2.Canny(gray, 100, 200)
        return edges
    
    else:
        return frame


def convert_to_grayscale(frame):
    """
    Convierte un frame a escala de grises
    
    Args:
        frame: Frame en color
    
    Returns:
        Frame en escala de grises
    """
    if frame is None:
        return None
    
    if len(frame.shape) == 2:
        # Ya está en escala de grises
        return frame
    
    # Convertir de RGB a escala de grises
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def clean_dataset(data_dir: str, valid_extensions: List[str] = None) -> dict:
    """
    Limpia el dataset eliminando archivos corruptos o inválidos.
    
    Args:
        data_dir: Directorio raíz del dataset.
        valid_extensions: Lista de extensiones válidas de imagen.
        
    Returns:
        Diccionario con estadísticas de limpieza.
    """
    if valid_extensions is None:
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    data_path = Path(data_dir)
    stats = {
        'total_files': 0,
        'valid_images': 0,
        'corrupted_images': 0,
        'invalid_extensions': 0,
        'removed_files': []
    }
    
    # Iterar sobre todos los archivos
    for file_path in data_path.rglob('*'):
        if file_path.is_file():
            stats['total_files'] += 1
            
            # Verificar extensión
            if file_path.suffix.lower() not in valid_extensions:
                stats['invalid_extensions'] += 1
                continue
            
            # Intentar cargar la imagen
            try:
                image = cv2.imread(str(file_path))
                if image is None:
                    stats['corrupted_images'] += 1
                    stats['removed_files'].append(str(file_path))
                    print(f"Imagen corrupta eliminada: {file_path}")
                    file_path.unlink()  # Eliminar archivo
                else:
                    stats['valid_images'] += 1
            except Exception as e:
                stats['corrupted_images'] += 1
                stats['removed_files'].append(str(file_path))
                print(f"Error con imagen {file_path}: {str(e)}")
                try:
                    file_path.unlink()
                except:
                    pass
    
    return stats


def balance_dataset(data_dir: str, target_samples: Optional[int] = None) -> dict:
    """
    Balancea el dataset asegurando que todas las clases tengan similar cantidad de muestras.
    
    Args:
        data_dir: Directorio raíz del dataset.
        target_samples: Número objetivo de muestras por clase. Si es None, usa el mínimo.
        
    Returns:
        Diccionario con información sobre el balanceo.
    """
    data_path = Path(data_dir)
    
    # Contar muestras por clase
    class_counts = {}
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*.jpg')) + 
                       list(class_dir.glob('*.jpeg')) + 
                       list(class_dir.glob('*.png')))
            class_counts[class_dir.name] = count
    
    if not class_counts:
        return {'error': 'No se encontraron clases en el dataset'}
    
    # Determinar target_samples
    if target_samples is None:
        target_samples = min(class_counts.values())
    
    info = {
        'class_counts_before': class_counts.copy(),
        'target_samples': target_samples,
        'class_counts_after': {},
        'actions': []
    }
    
    print(f"\nBalanceando dataset a {target_samples} muestras por clase...")
    for class_name, count in class_counts.items():
        info['class_counts_after'][class_name] = min(count, target_samples)
        if count > target_samples:
            info['actions'].append(f"{class_name}: reducido de {count} a {target_samples}")
        else:
            info['actions'].append(f"{class_name}: mantiene {count} muestras")
    
    return info


def validate_image_quality(image_path: str, min_size: Tuple[int, int] = (100, 100)) -> bool:
    """
    Valida la calidad de una imagen según criterios mínimos.
    
    Args:
        image_path: Ruta a la imagen.
        min_size: Tamaño mínimo aceptable (ancho, alto).
        
    Returns:
        True si la imagen cumple los criterios, False en caso contrario.
    """
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return False
        
        height, width = image.shape[:2]
        
        # Verificar tamaño mínimo
        if width < min_size[0] or height < min_size[1]:
            return False
        
        # Verificar que no sea una imagen completamente negra o blanca
        mean_intensity = np.mean(image)
        if mean_intensity < 5 or mean_intensity > 250:
            return False
        
        return True
    except:
        return False
