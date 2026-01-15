# 🚨 PROBLEMA CRÍTICO - Modelo v2 NO Funciona

**Fecha:** 15 de enero de 2026  
**Reportado por:** Daniel (Team Lead - UI/Integration)  
**Responsable:** Igor (AI/CNN Model)  
**Estado:** 🔴 BLOQUEANTE

---

## ❌ Problema Detectado

El modelo `modelo_seguridad_v2.h5` y `modelo_seguridad_v2.keras` **tienen el mismo error** que el modelo v1:

```
Layer "dense_4" expects 1 input(s), but it received 2 input tensors.
```

### Pruebas Realizadas

Ejecuté `test_modelo_v2.py` y ambos formatos fallan:
- ❌ `modelo_seguridad_v2.h5` - Error al cargar
- ❌ `modelo_seguridad_v2.keras` - Error al cargar

**Conclusión:** El modelo v2 NO es diferente al v1 en términos de arquitectura. Tiene el mismo defecto.

---

## 🔍 Análisis Técnico

### El Error Indica:

La capa `dense_4` está recibiendo **2 tensores de entrada** cuando solo espera **1 tensor**.

Esto típicamente ocurre cuando:
1. **Usas una capa de concatenación/merge antes de dense_4** pero no la defines correctamente
2. **Hay capas residuales (skip connections)** mal implementadas
3. **El modelo usa Functional API** con múltiples inputs que no se fusionaron correctamente

### Información del Modelo v2:

```python
Input shape: (None, 224, 224, 3)
Output shape: Debe ser (None, 4)  # 4 clases
```

---

## ✅ SOLUCIÓN REQUERIDA

Igor, necesitas **RECONSTRUIR** el modelo desde cero. Aquí está **EXACTAMENTE** lo que debes hacer:

### Opción 1: Modelo Secuencial Simple (RECOMENDADO)

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def crear_modelo_seguridad_v2():
    """
    Modelo CNN simple y funcional para detección de objetos de seguridad
    4 clases: arma, gorro, mascara, persona
    Input: 224x224x3 (RGB)
    """
    
    model = keras.Sequential([
        # Bloque 1: Convolucional
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Bloque 2: Convolucional
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Bloque 3: Convolucional
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Bloque 4: Convolucional
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
        
        # Capa de salida: 4 clases
        layers.Dense(4, activation='softmax')
    ])
    
    # Compilar
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Crear y guardar el modelo
model = crear_modelo_seguridad_v2()

# Ver resumen para verificar
model.summary()

# Guardar en AMBOS formatos
model.save('models/modelo_seguridad_v2.h5')  # Formato HDF5
model.save('models/modelo_seguridad_v2.keras')  # Formato Keras nativo

print("✅ Modelo v2 creado y guardado correctamente!")
```

### Opción 2: Transfer Learning con MobileNetV2 (MEJOR ACCURACY)

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

def crear_modelo_transfer_learning():
    """
    Modelo usando Transfer Learning con MobileNetV2
    Mayor accuracy pero requiere más recursos
    """
    
    # Cargar MobileNetV2 pre-entrenado (sin la capa superior)
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Congelar capas base
    base_model.trainable = False
    
    # Construir modelo
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),  # ⚠️ IMPORTANTE: Esto evita el error de múltiples tensors
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Usar esta opción si quieres mejor accuracy
model = crear_modelo_transfer_learning()
model.summary()
model.save('models/modelo_seguridad_v2.h5')
model.save('models/modelo_seguridad_v2.keras')
```

---

## ⚠️ REGLAS CRÍTICAS

1. **NO uses capas Concatenate, Add, Multiply sin GlobalAveragePooling o Flatten**
2. **Si usas Functional API, asegúrate de que SOLO un tensor llegue a cada Dense layer**
3. **Prueba el modelo ANTES de subirlo**:

```python
# Prueba rápida
import numpy as np
test_input = np.random.rand(1, 224, 224, 3)
prediction = model.predict(test_input)
print(f"Shape de predicción: {prediction.shape}")  # Debe ser (1, 4)
assert prediction.shape == (1, 4), "ERROR: Shape incorrecta!"
```

4. **Verifica que se carga correctamente**:

```python
# Cargar y probar
loaded_model = keras.models.load_model('models/modelo_seguridad_v2.h5')
test_prediction = loaded_model.predict(test_input)
print("✅ Modelo carga correctamente!")
```

---

## 📋 Checklist para Igor

Antes de subir el modelo v2 corregido, verifica:

- [ ] El modelo se crea sin errores
- [ ] `model.summary()` muestra arquitectura correcta
- [ ] El output shape es `(None, 4)`
- [ ] El modelo se guarda en ambos formatos (.h5 y .keras)
- [ ] El modelo se puede **cargar** sin errores
- [ ] Una predicción de prueba funciona
- [ ] Ejecutaste `test_modelo_v2.py` y pasó las pruebas
- [ ] Hiciste commit solo DESPUÉS de verificar todo lo anterior

---

## 🎯 Próximos Pasos

### Para Igor:
1. Leer este documento completo
2. Elegir Opción 1 (simple) u Opción 2 (transfer learning)
3. Crear el modelo siguiendo el código exacto
4. Probar con el script `test_modelo_v2.py`
5. Si pasa las pruebas → commit y push
6. Notificar al equipo en el chat

### Para Daniel (cuando Igor termine):
1. Actualizar `config.json` para usar `modelo_seguridad_v2.keras`
2. Probar en Streamlit UI
3. Hacer pruebas de integración completas
4. Documentar en README

---

## 📞 Contacto

Si tienes dudas sobre el error o la arquitectura, contacta:
- **Daniel** (daniel/ui-integration) - Integración y troubleshooting
- **D'Alessandro** (dalessandro/preprocessing) - Preprocesamiento de imágenes
- **Bruno** (bruno/logic-tests) - Testing y validación

**NO subas el modelo hasta que pase `test_modelo_v2.py` exitosamente.** ✅

---

**Actualizado:** 15/01/2026 00:10  
**Urgencia:** 🔴 Alta - Bloquea el avance del proyecto
