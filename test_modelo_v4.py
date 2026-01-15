"""
Script de prueba para el modelo de seguridad v4
Prueba ambas versiones (.h5 y .keras) con imágenes reales
"""
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import cv2
from src.preprocessing import preprocess_frame

def test_model(model_path, test_images_dir='data/raw'):
    """
    Prueba un modelo con imágenes de cada clase
    
    Args:
        model_path: Ruta al modelo (.h5 o .keras)
        test_images_dir: Directorio con las imágenes de prueba
    """
    print(f"\n{'='*60}")
    print(f"PROBANDO MODELO: {model_path}")
    print(f"{'='*60}\n")
    
    # 1. Cargar el modelo
    try:
        print("Cargando modelo...")
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ Modelo cargado exitosamente!")
        
        # Recompilar
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Mostrar información del modelo
        print(f"\n📊 Información del modelo:")
        print(f"   - Input shape: {model.input_shape}")
        print(f"   - Output shape: {model.output_shape}")
        print(f"   - Número de capas: {len(model.layers)}")
        
    except Exception as e:
        print(f"❌ ERROR al cargar modelo: {e}")
        return False
    
    # 2. Probar con imágenes de cada clase
    clases = ['arma', 'gorro', 'mascara', 'persona']
    print(f"\n🔍 Probando con imágenes reales de cada clase:\n")
    
    resultados_totales = {
        'correctas': 0,
        'incorrectas': 0,
        'total': 0
    }
    
    for clase in clases:
        clase_dir = Path(test_images_dir) / clase
        
        if not clase_dir.exists():
            print(f"⚠️  Directorio {clase} no encontrado")
            continue
        
        # Tomar las primeras 3 imágenes de cada clase
        imagenes = list(clase_dir.glob('*.jpg'))[:3]
        
        print(f"\n📁 Clase: {clase.upper()}")
        print(f"   Imágenes encontradas: {len(list(clase_dir.glob('*.jpg')))}")
        print(f"   Probando con: {len(imagenes)} imágenes")
        
        for img_path in imagenes:
            try:
                # Leer imagen
                frame = cv2.imread(str(img_path))
                if frame is None:
                    print(f"   ⚠️  No se pudo leer: {img_path.name}")
                    continue
                
                # Preprocesar (usando función de D'Alessandro)
                input_frame = preprocess_frame(frame)
                input_frame = np.expand_dims(input_frame, axis=0)
                
                # Predecir
                predictions = model.predict(input_frame, verbose=0)
                idx_predicho = np.argmax(predictions[0])
                confianza = predictions[0][idx_predicho]
                clase_predicha = clases[idx_predicho]
                
                # Verificar si es correcto
                es_correcto = clase_predicha == clase
                simbolo = "✅" if es_correcto else "❌"
                
                resultados_totales['total'] += 1
                if es_correcto:
                    resultados_totales['correctas'] += 1
                else:
                    resultados_totales['incorrectas'] += 1
                
                print(f"   {simbolo} {img_path.name:20} -> {clase_predicha:10} (confianza: {confianza:.2%})")
                
                # Mostrar todas las probabilidades si está mal clasificada
                if not es_correcto:
                    print(f"      Distribución: ", end="")
                    for i, c in enumerate(clases):
                        print(f"{c}: {predictions[0][i]:.2%} ", end="")
                    print()
                    
            except Exception as e:
                print(f"   ❌ Error procesando {img_path.name}: {e}")
    
    # 3. Resumen final
    print(f"\n{'='*60}")
    print(f"RESUMEN DE PRUEBAS")
    print(f"{'='*60}")
    print(f"Total imágenes probadas: {resultados_totales['total']}")
    print(f"✅ Correctas: {resultados_totales['correctas']}")
    print(f"❌ Incorrectas: {resultados_totales['incorrectas']}")
    
    if resultados_totales['total'] > 0:
        accuracy = (resultados_totales['correctas'] / resultados_totales['total']) * 100
        print(f"📊 Accuracy: {accuracy:.2f}%")
        
        if accuracy >= 75:
            print(f"✅ ¡Modelo funciona bien! (≥75%)")
            return True
        elif accuracy >= 50:
            print(f"⚠️  Modelo funciona pero tiene margen de mejora (50-75%)")
            return True
        else:
            print(f"❌ Modelo necesita más entrenamiento (<50%)")
            return False
    
    return False


if __name__ == "__main__":
    print("🔒 SISTEMA DE PRUEBA DE MODELO DE SEGURIDAD V4")
    print("Desarrollado por el equipo de IA/Seguridad\n")
    
    # Probar modelo v4 en formato .h5
    print("\n" + "="*60)
    print("TEST 1: Modelo v4 formato HDF5 (.h5)")
    print("="*60)
    resultado_h5 = test_model('models/modelo_seguridad_v4.h5')
    
    # Probar modelo v4 en formato .keras
    print("\n\n" + "="*60)
    print("TEST 2: Modelo v4 formato Keras (.keras)")
    print("="*60)
    resultado_keras = test_model('models/modelo_seguridad_v4.keras')
    
    # Resultado final
    print("\n\n" + "="*60)
    print("CONCLUSIÓN FINAL")
    print("="*60)
    
    if resultado_h5 and resultado_keras:
        print("✅ ¡Ambos modelos funcionan correctamente!")
        print("💡 Recomendación: Usar modelo_seguridad_v4.keras (formato más reciente)")
        print("\n📝 Próximos pasos:")
        print("   1. Actualizar config.json para usar modelo v4")
        print("   2. Probar en la interfaz Streamlit")
        print("   3. Hacer commit y push del nuevo modelo")
    elif resultado_h5:
        print("✅ Modelo .h5 funciona")
        print("⚠️  Modelo .keras tiene problemas")
        print("💡 Usar modelo_seguridad_v4.h5")
    elif resultado_keras:
        print("✅ Modelo .keras funciona")
        print("⚠️  Modelo .h5 tiene problemas")
        print("💡 Usar modelo_seguridad_v4.keras")
    else:
        print("❌ Ambos modelos tienen problemas")
        print("💡 Contactar a Igor para revisar los modelos")
