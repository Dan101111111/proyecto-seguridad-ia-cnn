"""
Script rápido para probar el sistema completo
Usa el modelo v2 (sin entrenar) para verificar que todo funciona
"""
import sys
sys.path.insert(0, '.')

print("="*70)
print("PRUEBA RÁPIDA DEL SISTEMA DE SEGURIDAD")
print("="*70)

# 1. Verificar que el modelo v2 carga
print("\n1️⃣ Probando carga del modelo v2...")
try:
    import tensorflow as tf
    model = tf.keras.models.load_model('models/modelo_seguridad_v2.keras', compile=False)
    print(f"   ✅ Modelo v2 carga correctamente")
    print(f"   📊 Input: {model.input_shape}, Output: {model.output_shape}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    exit(1)

# 2. Verificar preprocesamiento
print("\n2️⃣ Probando preprocesamiento...")
try:
    from src.preprocessing import preprocess_frame
    import cv2
    import numpy as np
    
    # Leer una imagen de prueba
    img = cv2.imread('data/raw/arma/arma_001.jpg')
    processed = preprocess_frame(img)
    print(f"   ✅ Preprocesamiento funciona")
    print(f"   📊 Shape procesada: {processed.shape}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    exit(1)

# 3. Verificar detección
print("\n3️⃣ Probando detección...")
try:
    from src.detector import detect_objects
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    detections = detect_objects(img, model, threshold=0.5)
    print(f"   ✅ Detección funciona")
    print(f"   📦 Detecciones: {detections}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    exit(1)

# 4. Verificar análisis de riesgo
print("\n4️⃣ Probando análisis de riesgo (Bruno)...")
try:
    from src.logic import check_security_risk
    
    risk = check_security_risk(detections)
    print(f"   ✅ Análisis de riesgo funciona")
    print(f"   ⚠️  Nivel: {risk.get('risk_level', 'N/A')}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    exit(1)

# 5. Verificar configuración
print("\n5️⃣ Probando carga de configuración...")
try:
    from src.utils import load_config
    
    config = load_config()
    print(f"   ✅ Configuración carga correctamente")
    print(f"   📋 Modelo configurado: {config['model']['path']}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    exit(1)

print("\n" + "="*70)
print("✅ TODAS LAS PRUEBAS PASARON")
print("="*70)
print("\n📝 Sistema funcionando correctamente!")
print("⚠️  NOTA: Modelo v2 NO está entrenado (accuracy ~25%)")
print("💡 Siguiente paso: Igor debe entrenar el modelo")
print("\n🚀 Para probar la UI: streamlit run ui/app.py")
print("="*70)
