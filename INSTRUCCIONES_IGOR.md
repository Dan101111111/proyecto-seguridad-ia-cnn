# 🎯 Instrucciones para Igor - Modelo de Seguridad v2

**Fecha:** 15 de enero de 2026  
**Responsable:** Igor (AI/CNN Model)  
**Estado:** 🟡 Modelo v2 funcional pero sin entrenar

---

## 📊 Resumen de la Situación

El modelo v2 que subiste tenía un error de arquitectura (mismo problema que v1). Daniel ya creó un modelo v2 **FUNCIONAL** con la arquitectura correcta, pero este modelo **NO está entrenado** y tiene accuracy ~25% (predicción aleatoria).

**Tu tarea:** Entrenar el modelo con las imágenes de `data/raw/`

---

## ✅ Paso 1: Verificar el Modelo Actual

Primero, confirma que el modelo carga correctamente:

```bash
# Activar entorno virtual
.venv\Scripts\Activate.ps1

# Ejecutar pruebas
python test_modelo_v2.py
```

**Resultado esperado:**

- ✅ Modelo carga sin errores
- ⚠️ Accuracy ~25% (modelo sin entrenar)

---

## 🔥 Paso 2: Entrenar el Modelo

Crea un archivo llamado `entrenar_modelo_v2.py` con este código:

```python
"""
Script para entrenar el modelo de seguridad v2
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("="*70)
print("ENTRENAMIENTO MODELO DE SEGURIDAD V2")
print("="*70)

# Configuración
BATCH_SIZE = 32
EPOCHS = 50  # Puedes aumentar a 100 si tienes tiempo
IMG_SIZE = (224, 224)
DATA_DIR = 'data/raw'

# 1. Cargar modelo sin entrenar
print("\n📂 Cargando modelo v2...")
model = keras.models.load_model('models/modelo_seguridad_v2.keras')
print("✅ Modelo cargado!")

# 2. Preparar datos con Data Augmentation
print("\n📊 Preparando datos...")
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

# 3. Generadores
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

print(f"\nClases detectadas: {train_generator.class_indices}")
print(f"Imágenes entrenamiento: {train_generator.samples}")
print(f"Imágenes validación: {validation_generator.samples}")

# 4. Entrenar
print(f"\n🔥 Entrenando por {EPOCHS} épocas...")
print("Esto tomará ~15-30 minutos...\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    verbose=1
)

# 5. Guardar modelo entrenado
print("\n💾 Guardando modelo entrenado...")
model.save('models/modelo_seguridad_v2.h5')
model.save('models/modelo_seguridad_v2.keras')
print("✅ Modelo guardado!")

# 6. Resultados
final_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print("\n" + "="*70)
print("RESULTADOS")
print("="*70)
print(f"Accuracy entrenamiento: {final_acc*100:.2f}%")
print(f"Accuracy validación: {final_val_acc*100:.2f}%")

if final_val_acc > 0.80:
    print("\n✅ ¡Excelente! Modelo listo (>80%)")
elif final_val_acc > 0.60:
    print("\n⚠️  Aceptable pero puede mejorar (60-80%)")
else:
    print("\n❌ Necesita más entrenamiento (<60%)")

print("\n🎯 Próximo paso: python test_modelo_v2.py")
```

**Ejecutar el entrenamiento:**

```bash
python entrenar_modelo_v2.py
```

---

## 🧪 Paso 3: Verificar el Modelo Entrenado

Una vez terminado el entrenamiento:

```bash
python test_modelo_v2.py
```

**Resultado esperado:**

- ✅ Accuracy > 60% (mínimo aceptable)
- ✅ Accuracy > 80% (ideal para producción)

---

## 🎨 Paso 4: Probar en la Interfaz Streamlit

```bash
streamlit run ui/app.py
```

Abre http://localhost:8502 y prueba:

1. **Detección en Imagen:**
   - Sube `data/raw/arma/arma_001.jpg`
   - Verifica que detecta "arma" con > 60% confianza
2. **Probar otras clases:**
   - Gorro: `data/raw/gorro/casco_001.jpg`
   - Máscara: `data/raw/mascara/mask_001.jpg`
   - Persona: `data/raw/persona/persona_001.jpg`

---

## 📋 Paso 5: Subir Cambios

Si el modelo funciona correctamente:

```bash
# Cambiar a tu rama
git checkout igor/cnn-model

# Agregar modelos entrenados
git add models/modelo_seguridad_v2.h5
git add models/modelo_seguridad_v2.keras
git add entrenar_modelo_v2.py

# Commit
git commit -m "Modelo v2 entrenado con accuracy del XX%"

# Push
git push origin igor/cnn-model

# Notificar a Daniel en el chat del equipo
```

---

## 📊 Métricas de Éxito

### Mínimo aceptable:

- ✅ Modelo carga sin errores
- ✅ Accuracy > 60%
- ✅ Detecciones coherentes en UI

### Ideal:

- ✅ Accuracy > 80%
- ✅ Confianza promedio > 70%
- ✅ Todas las clases balanceadas

---

## 🐛 Solución de Problemas

### Error: "No module named 'tensorflow'"

```bash
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Warning: "albumentations no disponible"

- ⚠️ Normal, no afecta el entrenamiento básico
- Opcional: `pip install albumentations`

### Entrenamiento muy lento

- Reduce EPOCHS a 30
- Aumenta BATCH_SIZE a 64 (si tienes RAM suficiente)

### Accuracy muy baja (<50%)

- Aumenta EPOCHS a 100
- Verifica que las imágenes son de buena calidad
- Revisa que las carpetas en `data/raw/` están correctas

---

## 📁 Estructura de Datos

```
data/raw/
├── arma/       60 imágenes → Riesgo ALTO
├── gorro/      30 imágenes → Riesgo MEDIO
├── mascara/    30 imágenes → Riesgo MEDIO-ALTO
└── persona/    60 imágenes → Riesgo BAJO
```

Total: 180 imágenes

- 80% entrenamiento = 144 imágenes
- 20% validación = 36 imágenes

---

## 🎯 Alternativa: Modelo con Transfer Learning

Si quieres mejor accuracy, usa Transfer Learning con MobileNetV2:

```python
python crear_modelo_v2_funcional.py
# Selecciona opción 2
```

Luego entrena ese modelo en lugar del simple.

---

## 📞 Ayuda

Si tienes problemas:

- **Daniel** - Integración y UI
- **D'Alessandro** - Preprocesamiento
- **Bruno** - Testing

---

## ✅ Checklist Final

Antes de notificar que está listo:

- [ ] `test_modelo_v2.py` pasa las pruebas
- [ ] Accuracy > 60%
- [ ] Streamlit UI muestra detecciones correctas
- [ ] Modelo guardado en ambos formatos (.h5 y .keras)
- [ ] Commit y push a rama igor/cnn-model
- [ ] Notificado al equipo

---

**Última actualización:** 15/01/2026  
**Prioridad:** 🔴 Alta - Bloquea avance del proyecto  
**Tiempo estimado:** 30-60 minutos (incluyendo entrenamiento)
