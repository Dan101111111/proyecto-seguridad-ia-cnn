"""
Script de prueba para preprocessing.py y utils.py
Autor: D'Alessandro
"""

import sys
import os
from pathlib import Path
import numpy as np

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocessing import (
    ImagePreprocessor, 
    DataAugmentor,
    preprocess_frame,
    resize_frame,
    normalize_frame,
    apply_filters,
    convert_to_grayscale,
    clean_dataset,
    balance_dataset,
    validate_image_quality
)

from utils import (
    setup_logging,
    save_image,
    load_config,
    save_config,
    get_timestamp,
    create_output_directory,
    format_detection_results,
    validate_input,
    create_directory_structure,
    get_class_names,
    count_samples_per_class,
    plot_class_distribution,
    format_time,
    get_device,
    print_system_info
)


def test_preprocessing():
    """Prueba las funciones de preprocesamiento."""
    print("\n" + "="*60)
    print("PRUEBAS DE PREPROCESSING.PY")
    print("="*60)
    
    # Verificar si hay imágenes en data/raw
    data_dir = Path(__file__).parent.parent / 'data' / 'raw'
    
    if not data_dir.exists():
        print(f"\n⚠️  El directorio {data_dir} no existe.")
        print("   Creando estructura de directorios de ejemplo...")
        data_dir.mkdir(parents=True, exist_ok=True)
        print("   Por favor, coloca imágenes en las siguientes carpetas:")
        print(f"   - {data_dir}/arma")
        print(f"   - {data_dir}/gorro")
        print(f"   - {data_dir}/mascara")
        print(f"   - {data_dir}/persona")
        return False
    
    # Buscar imágenes
    image_files = list(data_dir.rglob('*.jpg')) + list(data_dir.rglob('*.png')) + list(data_dir.rglob('*.jpeg'))
    
    if not image_files:
        print(f"\n⚠️  No se encontraron imágenes en {data_dir}")
        print("   Por favor, agrega imágenes para ejecutar las pruebas.")
        return False
    
    print(f"\n✓ Encontradas {len(image_files)} imágenes para probar")
    test_image = str(image_files[0])
    print(f"  Usando imagen de prueba: {Path(test_image).name}")
    
    # Test 1: ImagePreprocessor
    print("\n--- Test 1: ImagePreprocessor ---")
    try:
        preprocessor = ImagePreprocessor(target_size=(224, 224))
        
        # Cargar imagen
        image = preprocessor.load_image(test_image)
        if image is not None:
            print(f"✓ Imagen cargada: shape={image.shape}, dtype={image.dtype}")
            
            # Redimensionar
            resized = preprocessor.resize_image(image)
            print(f"✓ Imagen redimensionada: shape={resized.shape}")
            
            # Normalizar
            normalized = preprocessor.normalize_image(image)
            print(f"✓ Imagen normalizada: min={normalized.min():.3f}, max={normalized.max():.3f}")
            
            # Remover ruido
            denoised = preprocessor.remove_noise(image)
            print(f"✓ Ruido removido: shape={denoised.shape}")
            
            # Ajustar brillo/contraste
            adjusted = preprocessor.adjust_brightness_contrast(image, brightness=10, contrast=1.2)
            print(f"✓ Brillo/Contraste ajustado: shape={adjusted.shape}")
            
            # Pipeline completo
            processed = preprocessor.preprocess_image(test_image, normalize=True, remove_noise_flag=True)
            if processed is not None:
                print(f"✓ Pipeline completo: shape={processed.shape}")
        else:
            print("✗ Error al cargar la imagen")
            return False
            
    except Exception as e:
        print(f"✗ Error en ImagePreprocessor: {str(e)}")
        return False
    
    # Test 2: Funciones básicas
    print("\n--- Test 2: Funciones Básicas ---")
    try:
        # preprocess_frame
        frame_processed = preprocess_frame(image, target_size=(224, 224))
        print(f"✓ preprocess_frame: shape={frame_processed.shape}")
        
        # resize_frame
        frame_resized = resize_frame(image, 300, 300)
        print(f"✓ resize_frame: shape={frame_resized.shape}")
        
        # normalize_frame
        frame_normalized = normalize_frame(image)
        print(f"✓ normalize_frame: min={frame_normalized.min():.3f}, max={frame_normalized.max():.3f}")
        
        # apply_filters
        frame_blur = apply_filters(image, 'blur')
        print(f"✓ apply_filters (blur): shape={frame_blur.shape}")
        
        frame_sharpen = apply_filters(image, 'sharpen')
        print(f"✓ apply_filters (sharpen): shape={frame_sharpen.shape}")
        
        # convert_to_grayscale
        gray = convert_to_grayscale(image)
        print(f"✓ convert_to_grayscale: shape={gray.shape}")
        
    except Exception as e:
        print(f"✗ Error en funciones básicas: {str(e)}")
        return False
    
    # Test 3: Validación de calidad
    print("\n--- Test 3: Validación de Calidad ---")
    try:
        is_valid = validate_image_quality(test_image, min_size=(50, 50))
        print(f"✓ validate_image_quality: {is_valid}")
    except Exception as e:
        print(f"✗ Error en validación: {str(e)}")
    
    # Test 4: Limpieza de dataset
    print("\n--- Test 4: Análisis de Dataset ---")
    try:
        # Contar muestras si existen clases
        classes = [d for d in data_dir.iterdir() if d.is_dir()]
        if classes:
            print(f"✓ Clases encontradas: {[c.name for c in classes]}")
            
            stats = clean_dataset(str(data_dir))
            print(f"✓ Estadísticas de limpieza:")
            print(f"  - Total archivos: {stats['total_files']}")
            print(f"  - Imágenes válidas: {stats['valid_images']}")
            print(f"  - Imágenes corruptas: {stats['corrupted_images']}")
            
            balance_info = balance_dataset(str(data_dir))
            if 'error' not in balance_info:
                print(f"✓ Información de balance:")
                for class_name, count in balance_info['class_counts_before'].items():
                    print(f"  - {class_name}: {count} muestras")
        else:
            print("⚠️  No hay subdirectorios de clases. Estructura esperada:")
            print("   data/raw/arma/, data/raw/gorro/, etc.")
            
    except Exception as e:
        print(f"✗ Error en análisis de dataset: {str(e)}")
    
    return True


def test_utils():
    """Prueba las funciones de utilidades."""
    print("\n" + "="*60)
    print("PRUEBAS DE UTILS.PY")
    print("="*60)
    
    # Test 1: Logging
    print("\n--- Test 1: Sistema de Logging ---")
    try:
        logger = setup_logging(log_dir="logs", log_level="INFO")
        logger.info("Prueba de logging exitosa")
        print("✓ Sistema de logging configurado correctamente")
    except Exception as e:
        print(f"✗ Error en logging: {str(e)}")
    
    # Test 2: Configuración
    print("\n--- Test 2: Manejo de Configuración ---")
    try:
        # Crear configuración de prueba
        test_config = {
            "model": {
                "name": "CNN_Security",
                "input_size": [224, 224],
                "num_classes": 4
            },
            "training": {
                "batch_size": 32,
                "epochs": 50,
                "learning_rate": 0.001
            }
        }
        
        config_path = Path(__file__).parent / "test_config.json"
        save_config(test_config, str(config_path))
        print(f"✓ Configuración guardada en {config_path}")
        
        loaded_config = load_config(str(config_path))
        print(f"✓ Configuración cargada: {len(loaded_config)} secciones")
        
        # Limpiar archivo de prueba
        config_path.unlink()
        
    except Exception as e:
        print(f"✗ Error en configuración: {str(e)}")
    
    # Test 3: Directorios y archivos
    print("\n--- Test 3: Gestión de Directorios ---")
    try:
        timestamp = get_timestamp()
        print(f"✓ Timestamp: {timestamp}")
        
        test_dir = Path(__file__).parent / "test_output"
        created_dir = create_output_directory(str(test_dir))
        print(f"✓ Directorio creado: {created_dir}")
        
        # Limpiar
        test_dir.rmdir()
        
    except Exception as e:
        print(f"✗ Error en gestión de directorios: {str(e)}")
    
    # Test 4: Validación de entrada
    print("\n--- Test 4: Validación de Entrada ---")
    try:
        # Validar numpy array
        test_array = np.random.rand(224, 224, 3)
        is_valid = validate_input(test_array)
        print(f"✓ Validación de numpy array: {is_valid}")
        
        # Validar None
        is_valid_none = validate_input(None)
        print(f"✓ Validación de None: {is_valid_none}")
        
    except Exception as e:
        print(f"✗ Error en validación: {str(e)}")
    
    # Test 5: Formato de detecciones
    print("\n--- Test 5: Formato de Resultados ---")
    try:
        test_detections = [
            {"class": "mascara", "confidence": 0.95, "bbox": [10, 20, 100, 150]},
            {"class": "persona", "confidence": 0.87, "bbox": [50, 60, 200, 300]}
        ]
        
        formatted = format_detection_results(test_detections)
        print("✓ Resultados formateados:")
        print(formatted)
        
    except Exception as e:
        print(f"✗ Error en formato de resultados: {str(e)}")
    
    # Test 6: Métricas y tiempo
    print("\n--- Test 6: Utilidades Generales ---")
    try:
        time_str = format_time(3725)  # 1h 2m 5s
        print(f"✓ Formato de tiempo (3725s): {time_str}")
        
        device = get_device()
        print(f"✓ Dispositivo detectado: {device}")
        
    except Exception as e:
        print(f"✗ Error en utilidades: {str(e)}")
    
    # Test 7: Información del sistema
    print("\n--- Test 7: Información del Sistema ---")
    try:
        print_system_info()
        print("✓ Información del sistema obtenida")
    except Exception as e:
        print(f"✗ Error al obtener info del sistema: {str(e)}")
    
    return True


def test_dataset_analysis():
    """Prueba funciones de análisis de dataset si existen imágenes."""
    print("\n" + "="*60)
    print("ANÁLISIS DE DATASET")
    print("="*60)
    
    data_dir = Path(__file__).parent.parent / 'data' / 'raw'
    
    if not data_dir.exists():
        print("\n⚠️  No hay dataset para analizar")
        return False
    
    try:
        # Obtener nombres de clases
        class_names = get_class_names(str(data_dir))
        if class_names:
            print(f"\n✓ Clases en el dataset: {class_names}")
            
            # Contar muestras por clase
            counts = count_samples_per_class(str(data_dir))
            print(f"✓ Conteo por clase:")
            for class_name, count in counts.items():
                print(f"  - {class_name}: {count} imágenes")
            
            # Visualizar distribución (sin mostrar)
            # plot_class_distribution(counts, save_path="results/class_distribution.png")
            
            return True
        else:
            print("\n⚠️  No se encontraron clases en el dataset")
            return False
            
    except Exception as e:
        print(f"✗ Error en análisis de dataset: {str(e)}")
        return False


def main():
    """Función principal para ejecutar todas las pruebas."""
    print("\n" + "="*60)
    print("INICIANDO PRUEBAS DE PREPROCESSING Y UTILS")
    print("="*60)
    
    # Pruebas de preprocessing
    preprocessing_ok = test_preprocessing()
    
    # Pruebas de utils
    utils_ok = test_utils()
    
    # Análisis de dataset
    dataset_ok = test_dataset_analysis()
    
    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN DE PRUEBAS")
    print("="*60)
    print(f"Preprocessing: {'✓ PASÓ' if preprocessing_ok else '⚠️  REQUIERE DATOS'}")
    print(f"Utils: {'✓ PASÓ' if utils_ok else '✗ FALLÓ'}")
    print(f"Dataset Analysis: {'✓ PASÓ' if dataset_ok else '⚠️  SIN DATASET'}")
    
    if not preprocessing_ok:
        print("\n📝 INSTRUCCIONES:")
        print("1. Coloca tus imágenes en:")
        print("   - data/raw/arma/")
        print("   - data/raw/gorro/")
        print("   - data/raw/mascara/")
        print("   - data/raw/persona/")
        print("2. Vuelve a ejecutar este script")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
