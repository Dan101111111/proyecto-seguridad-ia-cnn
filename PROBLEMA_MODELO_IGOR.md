# ⚠️ PROBLEMA CON EL MODELO - ATENCIÓN IGOR

## 🔴 Problema Detectado

El modelo `models/modelo_seguridad_v1.h5` **NO se puede cargar** correctamente. Al intentar cargarlo con TensorFlow, se produce el siguiente error:

```
Error al cargar el modelo: Layer "dense" expects 1 input(s), but it received 2 input tensors. 
Inputs received: [<KerasTensor shape=(None, 7, 7, 1280), dtype=float32, sparse=False, ragged=False, name=keras_tensor_321>, 
<KerasTensor shape=(None, 7, 7, 1280), dtype=float32, sparse=False, ragged=False, name=keras_tensor_322>]
```

## 📋 Origen del Problema

Este error indica que hay un **problema de incompatibilidad en la arquitectura del modelo**:

1. **Problema con la capa Dense**: La capa densa espera 1 tensor de entrada, pero está recibiendo 2 tensores
2. **Posible causa**: 
   - El modelo fue entrenado con una versión diferente de TensorFlow/Keras
   - Hay un error en la definición de la arquitectura del modelo
   - Problema al guardar el modelo (posiblemente con capas custom o concatenación incorrecta)

## 🛠️ Cambios Realizados para Mitigar el Problema

### 1. Modificación en `src/detector.py`

Se actualizó la función `load_model()` para intentar cargar con mayor compatibilidad:

```python
def load_model(model_path='models/modelo_seguridad_v1.h5'):
    """ Carga el modelo entrenado por Igor """
    try:
        # Intentar cargar con compile=False para evitar problemas de compatibilidad
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Modelo cargado correctamente.")
        
        # Recompilar el modelo con configuración básica
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        print("\n💡 SUGERENCIAS:")
        print("1. Verifica que el archivo del modelo sea compatible con TensorFlow 2.x")
        print("2. Regenera el modelo usando el mismo TensorFlow que tienes instalado")
        print("3. Contacta a Igor para que verifique la arquitectura del modelo")
        return None
```

### 2. Modelo Temporal de Prueba

Se creó un script `create_test_model.py` que genera un modelo temporal **NO ENTRENADO** solo para testing de la UI:

- **Archivo**: `models/modelo_seguridad_v1_TEMP.h5`
- **Propósito**: Permitir probar la interfaz mientras se corrige el modelo original
- **Estado**: Funcional pero con predicciones aleatorias (no está entrenado)

### 3. Actualización de `config.json`

Se cambió temporalmente la ruta del modelo:

```json
"model": {
  "path": "models/modelo_seguridad_v1_TEMP.h5",
  ...
}
```

## ✅ Estado Actual del Proyecto

- ✅ **Interfaz de usuario**: 100% funcional
- ✅ **Sistema de detección**: Funcionando con modelo temporal
- ✅ **Integración completa**: Todos los módulos del equipo integrados correctamente
- ⚠️ **Modelo original**: NO funcional - requiere corrección

## 🎯 Acciones Requeridas para Igor

### Opción 1: Regenerar el Modelo (RECOMENDADO)

Crea un nuevo modelo con esta estructura compatible:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Crear modelo compatible
model = keras.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')  # 4 clases: arma, gorro, mascara, persona
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# IMPORTANTE: Entrenar el modelo aquí...
# model.fit(...)

# Guardar correctamente
model.save('models/modelo_seguridad_v1.h5')
```

### Opción 2: Verificar Versión de TensorFlow

El proyecto usa actualmente:
- **TensorFlow**: ~2.13.0 o superior
- **Python**: 3.13.7

Asegúrate de usar la misma versión al entrenar y guardar el modelo.

### Opción 3: Exportar en Formato Compatible

Si ya tienes un modelo entrenado en otro formato, considera:

1. **SavedModel format** (recomendado):
   ```python
   model.save('models/modelo_seguridad_v1/')  # Sin .h5
   ```

2. **Formato Keras nativo** (.keras):
   ```python
   model.save('models/modelo_seguridad_v1.keras')
   ```

## 📊 Especificaciones del Modelo Esperado

### Entrada
- **Shape**: (224, 224, 3)
- **Tipo**: Imágenes RGB normalizadas [0, 1]

### Salida
- **Clases**: 4 (arma, gorro, mascara, persona)
- **Activación**: Softmax
- **Shape**: (4,)

### Preprocesamiento
Las imágenes se preprocesan usando la función `preprocess_frame()` de D'Alessandro que:
1. Redimensiona a 224x224
2. Normaliza a rango [0, 1]

## 🔗 Integración

Una vez corrijas el modelo:

1. Reemplaza `models/modelo_seguridad_v1.h5` con el modelo corregido
2. Actualiza `config.json`:
   ```json
   "path": "models/modelo_seguridad_v1.h5"
   ```
3. Reinicia la aplicación Streamlit
4. ¡Debería funcionar correctamente!

## 📞 Contacto

Si necesitas ayuda o tienes dudas:
- Coordina con **Daniel** (UI Integration)
- Revisa con **D'Alessandro** (Preprocessing) sobre el formato de entrada esperado
- Consulta con **Bruno** (Logic) sobre las clases y predicciones

---

**Última actualización**: 14 de enero de 2026  
**Estado**: Modelo original NO funcional - Usando modelo temporal de prueba  
**Urgencia**: Alta - Bloquea el testing completo del sistema
