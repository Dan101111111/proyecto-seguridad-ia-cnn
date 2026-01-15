"""
Script para crear modelo v4 con arquitectura CORRECTA y compatible
Ejecutar: python crear_modelo_v4_funcional.py
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print("="*70)
print("CREACIÓN DE MODELO DE SEGURIDAD V4 - VERSIÓN FUNCIONAL")
print("="*70)

# Opción del usuario
print("\nSelecciona el tipo de modelo:")
print("1. Modelo CNN Simple (rápido, menor accuracy)")
print("2. Transfer Learning MobileNetV2 (más lento, mejor accuracy) [RECOMENDADO]")

opcion = input("\nIngresa 1 o 2 [default: 2]: ").strip() or "2"

if opcion == "2":
    print("\n🔄 Creando modelo con Transfer Learning (MobileNetV2)...")
    
    # Cargar base pre-entrenada
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Congelar capas base
    base_model.trainable = False
    
    # Construir modelo
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),  # Crítico: convierte múltiples tensors en uno solo
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(4, activation='softmax')  # 4 clases
    ], name='modelo_seguridad_v4_transfer')
    
else:
    print("\n🔄 Creando modelo CNN Simple...")
    
    model = keras.Sequential([
        # Bloque 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Bloque 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Bloque 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Bloque 4
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Aplanamiento
        layers.Flatten(),
        
        # Capas densas
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        # Salida: 4 clases
        layers.Dense(4, activation='softmax')
    ], name='modelo_seguridad_v4_simple')

# Compilar
print("🔧 Compilando modelo...")
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Mostrar resumen
print("\n" + "="*70)
print("ARQUITECTURA DEL MODELO V4")
print("="*70)
model.summary()

# Verificación 1: Probar predicción
print("\n" + "="*70)
print("VERIFICACIÓN 1: Prueba de Predicción")
print("="*70)

test_input = np.random.rand(1, 224, 224, 3).astype('float32')
print(f"Input shape: {test_input.shape}")

try:
    prediction = model.predict(test_input, verbose=0)
    print(f"✅ Predicción exitosa!")
    print(f"   Output shape: {prediction.shape}")
    print(f"   Suma de probabilidades: {prediction.sum():.4f} (debe ser ~1.0)")
    
    assert prediction.shape == (1, 4), f"❌ ERROR: Shape incorrecta! Esperado (1, 4), obtenido {prediction.shape}"
    assert abs(prediction.sum() - 1.0) < 0.01, "❌ ERROR: Las probabilidades no suman 1!"
    
    print("✅ Todas las verificaciones pasaron!")
    
except Exception as e:
    print(f"❌ ERROR en predicción: {e}")
    print("\n⚠️  NO GUARDAR ESTE MODELO - Tiene errores!")
    exit(1)

# Guardar modelos
print("\n" + "="*70)
print("GUARDANDO MODELOS V4")
print("="*70)

try:
    # Guardar formato HDF5
    print("💾 Guardando modelo_seguridad_v4.h5...")
    model.save('models/modelo_seguridad_v4.h5')
    print("✅ Guardado exitosamente!")
    
    # Guardar formato Keras nativo
    print("💾 Guardando modelo_seguridad_v4.keras...")
    model.save('models/modelo_seguridad_v4.keras')
    print("✅ Guardado exitosamente!")
    
except Exception as e:
    print(f"❌ ERROR al guardar: {e}")
    exit(1)

# Verificación 2: Cargar y probar
print("\n" + "="*70)
print("VERIFICACIÓN 2: Carga de Modelos Guardados")
print("="*70)

try:
    # Probar .h5
    print("🔄 Cargando modelo_seguridad_v4.h5...")
    loaded_h5 = keras.models.load_model('models/modelo_seguridad_v4.h5', compile=False)
    loaded_h5.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    pred_h5 = loaded_h5.predict(test_input, verbose=0)
    print(f"✅ Modelo .h5 carga correctamente!")
    print(f"   Shape: {pred_h5.shape}")
    
    # Probar .keras
    print("\n🔄 Cargando modelo_seguridad_v4.keras...")
    loaded_keras = keras.models.load_model('models/modelo_seguridad_v4.keras', compile=False)
    loaded_keras.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    pred_keras = loaded_keras.predict(test_input, verbose=0)
    print(f"✅ Modelo .keras carga correctamente!")
    print(f"   Shape: {pred_keras.shape}")
    
except Exception as e:
    print(f"❌ ERROR al cargar modelo: {e}")
    print("\n⚠️  El modelo tiene problemas - NO HACER COMMIT!")
    exit(1)

# Verificación 3: Probar con script de prueba
print("\n" + "="*70)
print("VERIFICACIÓN 3: Listo para Pruebas")
print("="*70)
print("Puedes probar con: python test_modelo_v4.py")

# Resumen final
print("\n" + "="*70)
print("✅ ¡MODELO V4 CREADO EXITOSAMENTE!")
print("="*70)
print("\n📁 Archivos generados:")
print("   - models/modelo_seguridad_v4.h5")
print("   - models/modelo_seguridad_v4.keras")

print("\n📋 Características del modelo:")
if opcion == "2":
    print("   ✅ Transfer Learning con MobileNetV2")
    print("   ✅ Mejor accuracy esperado")
    print("   ✅ Pesos pre-entrenados en ImageNet")
else:
    print("   ✅ CNN Simple")
    print("   ✅ Entrenamiento más rápido")

print("\n🎯 Próximos pasos:")
print("   1. Actualizar config.json para usar modelo v4")
print("   2. Ejecutar: python test_modelo_v4.py")
print("   3. Probar en Streamlit: streamlit run ui/app.py")

print("\n" + "="*70)
