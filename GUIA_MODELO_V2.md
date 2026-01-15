# ✅ GUÍA: Modelo v2 Funcional - Próximos Pasos

**Fecha:** 15 de enero de 2026  
**Estado:** 🟢 Modelo v2 carga correctamente (arquitectura arreglada)  
**Pendiente:** 🟡 Igor necesita ENTRENAR el modelo

---

## 📊 Estado Actual del Proyecto

### ✅ Lo que YA funciona:

1. **Arquitectura del modelo v2**
   - ✅ Se carga sin errores
   - ✅ Input correcto: (224, 224, 3)
   - ✅ Output correcto: (4 clases)
   - ✅ Disponible en ambos formatos: `.h5` y `.keras`

2. **Integración completa**
   - ✅ UI Streamlit funcionando
   - ✅ Preprocesamiento (D'Alessandro) integrado
   - ✅ Lógica de seguridad (Bruno) integrada
   - ✅ Configuración actualizada a modelo v2

3. **Scripts de prueba**
   - ✅ `test_modelo_v2.py` - Prueba con imágenes reales
   - ✅ `crear_modelo_v2_funcional.py` - Crea modelo funcional

### 🟡 Lo que falta:

**Igor necesita ENTRENAR el modelo v2** con las imágenes de `data/raw/`

---

## 📋 Instrucciones para Igor

### Problema Identificado:

El modelo v2 actual es solo una arquitectura vacía (sin entrenar). Por eso:
- ✅ Se carga correctamente (arquitectura OK)
- ❌ Accuracy ~25% (predicción aleatoria - modelo no entrenado)

### Solución:

Necesitas **entrenar** el modelo usando las imágenes en `data/raw/`:

```
data/raw/
├── arma/       (60 imágenes)
├── gorro/      (30 imágenes)
├── mascara/    (30 imágenes)
└── persona/    (60 imágenes)
```

### Script de Entrenamiento (crear_entrenamiento_v2.py):

```python
"""
Script para entrenar el modelo de seguridad v2
Igor: Ejecuta este script para entrenar el modelo
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

print("="*70)
print("ENTRENAMIENTO MODELO DE SEGURIDAD V2")
print("="*70)

# 1. Configuración
BATCH_SIZE = 32
EPOCHS = 50  # Ajustar según tiempo disponible
IMG_SIZE = (224, 224)
DATA_DIR = 'data/raw'

# 2. Cargar modelo sin entrenar
print("\n📂 Cargando modelo v2 sin entrenar...")
model = keras.models.load_model('models/modelo_seguridad_v2.keras')

# 3. Preparar datos con Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% train, 20% validation
)

# 4. Generadores de datos
print("\n📊 Preparando datos de entrenamiento...")
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print(f"\n   Clases detectadas: {train_generator.class_indices}")
print(f"   Total imágenes entrenamiento: {train_generator.samples}")
print(f"   Total imágenes validación: {validation_generator.samples}")

# 5. Entrenar
print(f"\n🔥 Entrenando modelo por {EPOCHS} épocas...")
print("   Esto puede tomar varios minutos...\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    verbose=1
)

# 6. Guardar modelo entrenado
print("\n💾 Guardando modelo entrenado...")
model.save('models/modelo_seguridad_v2.h5')
model.save('models/modelo_seguridad_v2.keras')
print("✅ Modelo guardado exitosamente!")

# 7. Mostrar resultados
final_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print("\n" + "="*70)
print("RESULTADOS DEL ENTRENAMIENTO")
print("="*70)
print(f"Accuracy entrenamiento: {final_acc*100:.2f}%")
print(f"Accuracy validación: {final_val_acc*100:.2f}%")

if final_val_acc > 0.80:
    print("\n✅ ¡Excelente! Modelo listo para producción (>80%)")
elif final_val_acc > 0.60:
    print("\n⚠️  Modelo aceptable pero puede mejorar (60-80%)")
    print("💡 Sugerencia: Aumentar épocas o conseguir más datos")
else:
    print("\n❌ Modelo necesita más entrenamiento (<60%)")
    print("💡 Sugerencias:")
    print("   - Aumentar épocas a 100+")
    print("   - Conseguir más imágenes de entrenamiento")
    print("   - Revisar calidad de las imágenes")

print("\n🎯 Próximo paso: Ejecutar python test_modelo_v2.py")
```

---

## 🧪 Cómo Probar el Modelo Entrenado

### 1. Probar con script de prueba:

```bash
python test_modelo_v2.py
```

**Resultado esperado:**
- ✅ Ambos modelos cargan correctamente
- ✅ Accuracy > 60% (mínimo aceptable)
- ✅ Accuracy > 80% (ideal para producción)

### 2. Probar con Streamlit UI:

```bash
streamlit run ui/app.py
```

**Flujo de prueba:**
1. Subir una imagen de `data/raw/arma/arma_001.jpg`
2. Verificar que detecta "arma" con confianza > 60%
3. Ver análisis de riesgo de Bruno
4. Probar con imágenes de las otras clases

---

## 📊 Métricas de Éxito

### Mínimo aceptable:
- ✅ Modelo carga sin errores
- ✅ Accuracy validación > 60%
- ✅ UI muestra predicciones coherentes

### Ideal para producción:
- ✅ Accuracy validación > 80%
- ✅ Confianza promedio > 70%
- ✅ Todas las clases bien balanceadas

---

## 🎯 Plan de Acción

### Para Igor (URGENTE):

1. **Leer** este documento completo
2. **Crear** script de entrenamiento (copiar código arriba)
3. **Ejecutar** entrenamiento:
   ```bash
   python crear_entrenamiento_v2.py
   ```
4. **Esperar** ~15-30 minutos (según tu hardware)
5. **Verificar** con `test_modelo_v2.py`
6. Si accuracy > 60% → **Commit y push**
7. **Notificar** a Daniel que el modelo está entrenado

### Para Daniel (cuando Igor termine):

1. **Pull** cambios de Igor
2. **Probar** en Streamlit UI
3. **Documentar** accuracy final en README
4. **Hacer pruebas** de integración completas
5. **Preparar** demo para presentación

---

## 📝 Checklist Final

Antes de dar por terminado el modelo v2:

- [ ] Modelo se carga sin errores ✅ (YA HECHO)
- [ ] Modelo está entrenado (Igor)
- [ ] Accuracy validación > 60% (Igor)
- [ ] Test con imágenes reales pasa (Igor)
- [ ] Streamlit UI funciona correctamente (Daniel)
- [ ] Análisis de riesgo de Bruno integrado ✅ (YA HECHO)
- [ ] README actualizado con accuracy final
- [ ] Commit y push a rama igor/cnn-model
- [ ] Merge a main después de revisión del equipo

---

## 🔗 Archivos Relevantes

- **Modelo actual:** [models/modelo_seguridad_v2.keras](models/modelo_seguridad_v2.keras)
- **Configuración:** [config.json](config.json)
- **Script de prueba:** [test_modelo_v2.py](test_modelo_v2.py)
- **UI Streamlit:** [ui/app.py](ui/app.py)
- **Datos de entrenamiento:** [data/raw/](data/raw/)

---

## 📞 Contactos

- **Igor** (igor/cnn-model) - Entrenamiento del modelo
- **Daniel** (daniel/ui-integration) - UI y testing
- **D'Alessandro** (dalessandro/preprocessing) - Preprocesamiento
- **Bruno** (bruno/logic-tests) - Lógica de seguridad
- **Mario** (mario/data-validation) - Validación de datos

---

**Última actualización:** 15/01/2026 00:15  
**Estado:** 🟡 Esperando entrenamiento de Igor
