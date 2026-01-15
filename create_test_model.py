"""
Script para crear un modelo CNN de prueba temporal
Este modelo es solo para testing de la interfaz, NO para producción
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_test_model():
    """Crea un modelo simple de prueba"""
    
    model = keras.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4, activation='softmax')  # 4 clases
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    print("Creando modelo de prueba temporal...")
    model = create_test_model()
    
    # Guardar modelo
    model.save('models/modelo_seguridad_v1_TEMP.h5')
    print("✅ Modelo de prueba guardado en: models/modelo_seguridad_v1_TEMP.h5")
    print("\n⚠️ IMPORTANTE: Este es un modelo NO entrenado, solo para probar la UI")
    print("   Para usar este modelo temporal, actualiza config.json:")
    print('   "path": "models/modelo_seguridad_v1_TEMP.h5"')
