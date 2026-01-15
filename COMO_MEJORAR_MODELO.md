# 🎯 Guía para Mejorar la Precisión del Modelo v4

## 📊 Estado Actual
- **Accuracy validación**: 97.22% ✅
- **Accuracy pruebas reales**: 83.33%
- **Dataset**: 180 imágenes (60 arma, 30 gorro, 30 mascara, 60 persona)
- **Problemas detectados**: Clase "mascara" confunde con "gorro" (2/3 errores)

---

## 🚀 Estrategias para Mejorar la Precisión

### 1. **Conseguir MÁS DATOS** (⭐ MÁS IMPORTANTE)
La forma más efectiva de mejorar el modelo es aumentar el dataset:

#### 🎯 Objetivo Recomendado:
- **Mínimo**: 500 imágenes por clase (2,000 totales)
- **Ideal**: 1,000+ imágenes por clase (4,000+ totales)

#### 📸 Dónde conseguir datos:
```
Opción 1: Descargar datasets públicos
- Kaggle: https://www.kaggle.com/datasets
- Roboflow Universe: https://universe.roboflow.com/
- Google Dataset Search: https://datasetsearch.research.google.com/

Opción 2: Búsqueda de imágenes (usar con cuidado por derechos)
- Google Images (con filtro de licencia libre)
- Unsplash, Pexels (imágenes libres)
- Flickr (con licencia Creative Commons)

Opción 3: Data Augmentation automático
- Ya está implementado en entrenar_modelo_v4.py
- Genera variaciones automáticamente durante entrenamiento
```

#### 📁 Organización de nuevos datos:
```
data/raw/
├── arma/         [añadir hasta 500+ imágenes]
├── gorro/        [añadir hasta 500+ imágenes]
├── mascara/      [PRIORITARIO: solo 30 imágenes, añadir 470+]
└── persona/      [añadir hasta 500+ imágenes]
```

**⚠️ CRÍTICO**: La clase "mascara" solo tiene 30 imágenes, por eso falla. Necesita mínimo 200-300 más.

---

### 2. **Ajustar Hiperparámetros del Entrenamiento**

Edita `entrenar_modelo_v4.py`:

#### 🔧 Opciones a probar:

```python
# CONFIGURACIÓN ACTUAL (líneas 15-19)
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# PARA DATASETS PEQUEÑOS (<500 imágenes totales):
BATCH_SIZE = 16           # Reduce a 16 o 8
EPOCHS = 150              # Aumenta épocas
LEARNING_RATE = 0.0001    # Learning rate más bajo

# PARA DATASETS GRANDES (>1000 imágenes totales):
BATCH_SIZE = 64           # Aumenta batch size
EPOCHS = 50-75            # Menos épocas necesarias
LEARNING_RATE = 0.001     # Mantener o aumentar a 0.002
```

---

### 3. **Mejorar Data Augmentation**

Edita `entrenar_modelo_v4.py` (líneas 45-57):

#### Aumentar diversidad:
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,        # Aumentar de 30 a 40
    width_shift_range=0.3,    # Aumentar de 0.2 a 0.3
    height_shift_range=0.3,   # Aumentar de 0.2 a 0.3
    shear_range=0.3,          # Aumentar de 0.2 a 0.3
    zoom_range=0.3,           # Aumentar de 0.2 a 0.3
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  # NUEVO: Variación de brillo
    fill_mode='nearest',
    validation_split=0.2
)
```

---

### 4. **Descongelar Capas de MobileNetV2** (Avanzado)

Para datasets grandes (>500 imágenes/clase), puedes entrenar capas de MobileNetV2:

#### Crear nuevo script `entrenar_modelo_v4_avanzado.py`:

```python
# Cargar modelo v4
model = keras.models.load_model('models/modelo_seguridad_v4.keras')

# Descongelar las últimas 20 capas de MobileNetV2
base_model = model.layers[0]  # MobileNetV2
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Re-compilar con learning rate MUY bajo
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # 0.00001
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar 30-50 épocas más
# ... mismo código de entrenamiento
```

**⚠️ SOLO hacer esto si tienes >500 imágenes por clase**

---

### 5. **Usar Validación Cruzada** (K-Fold)

Para datasets pequeños, divide los datos en K partes y entrena K veces:

```python
# Ejemplo: entrenar 5 modelos diferentes
# Tomar el mejor o promediar sus predicciones
# Esto aprovecha mejor los datos limitados
```

---

### 6. **Balancear las Clases**

Actualmente:
- arma: 60 (33%)
- gorro: 30 (17%)
- mascara: 30 (17%)  ⚠️ DESBALANCEADO
- persona: 60 (33%)

#### Soluciones:
```python
# Opción A: Class weights (en entrenar_modelo_v4.py)
class_weights = {
    0: 1.0,  # arma (60 imágenes)
    1: 2.0,  # gorro (30 imágenes) - peso doble
    2: 2.0,  # mascara (30 imágenes) - peso doble
    3: 1.0   # persona (60 imágenes)
}

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    class_weight=class_weights,  # AÑADIR ESTO
    verbose=1
)
```

---

## 📋 Plan de Acción Recomendado

### 🥇 **PRIORIDAD 1: Más datos para "mascara"**
```bash
# 1. Descargar mínimo 200 imágenes de máscaras/mascarillas
# 2. Guardar en data/raw/mascara/
# 3. Nombres: mascara_031.jpg, mascara_032.jpg, etc.
```

### 🥈 **PRIORIDAD 2: Balancear dataset completo**
```bash
# Conseguir imágenes hasta que cada clase tenga:
# - Mínimo: 200 imágenes/clase (800 totales)
# - Ideal: 500 imágenes/clase (2,000 totales)
```

### 🥉 **PRIORIDAD 3: Re-entrenar con más datos**
```bash
python entrenar_modelo_v4.py
# Con más datos, la accuracy mejorará significativamente
```

### 🏅 **PRIORIDAD 4: Ajustar hiperparámetros**
```bash
# Probar diferentes configuraciones:
# - BATCH_SIZE: 8, 16, 32, 64
# - LEARNING_RATE: 0.0001, 0.001, 0.002
# - EPOCHS: 100, 150, 200
```

---

## 📊 Monitoreo de Mejoras

### Métricas a observar:
1. **Accuracy validación**: Debe ser >90%
2. **Accuracy entrenamiento**: Debe ser similar a validación (±5%)
3. **Confusion Matrix**: Ver qué clases se confunden
4. **Per-class accuracy**: Cada clase debe tener >80%

### Script para analizar resultados:
```python
# Crear analizar_resultados.py
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ... evaluar modelo con validation_generator
# ... mostrar matriz de confusión
# ... mostrar accuracy por clase
```

---

## 🎯 Expectativas Realistas

| Dataset Size | Accuracy Esperada |
|--------------|-------------------|
| <200 imágenes totales | 60-75% |
| 200-500 imágenes totales | 75-85% |
| 500-1000 imágenes totales | 85-92% |
| 1000-2000 imágenes totales | 92-96% |
| >2000 imágenes totales | 95-99% |

**Estado actual**: 180 imágenes → 83.33% accuracy ✅ (dentro de lo esperado)

---

## 🚨 Errores Comunes a Evitar

❌ **NO** entrenar con pocas imágenes esperando 95%+ accuracy
❌ **NO** aumentar EPOCHS a 500+ sin más datos (overfitting)
❌ **NO** descongelar MobileNetV2 con <500 imágenes/clase
❌ **NO** ignorar el desbalance de clases
❌ **NO** usar imágenes de baja calidad o irrelevantes

✅ **SÍ** conseguir más datos de calidad
✅ **SÍ** validar con imágenes reales no vistas
✅ **SÍ** monitorear overfitting (train vs val accuracy)
✅ **SÍ** usar data augmentation
✅ **SÍ** hacer backups antes de experimentar

---

## 🔄 Flujo de Trabajo Iterativo

```bash
# Ciclo de mejora continua:

1. Conseguir más datos → data/raw/
2. Entrenar modelo → python entrenar_modelo_v4.py
3. Evaluar resultados → python test_modelo_v4.py
4. Probar en UI → streamlit run ui/app.py
5. Identificar errores → anotar qué clases fallan
6. Repetir desde paso 1
```

---

## 📞 Recursos Útiles

### Datasets recomendados:
- **Armas**: Buscar "gun detection dataset" en Kaggle
- **Gorros/Cascos**: "helmet detection dataset", "PPE detection"
- **Máscaras**: "face mask detection dataset" (COVID-19)
- **Personas**: "person detection dataset", "COCO dataset"

### Herramientas:
- **Label Studio**: Para etiquetar imágenes propias
- **Roboflow**: Para procesamiento de datasets
- **Albumentations**: Augmentation avanzado (ya opcional en el código)

---

## ✅ Checklist Antes de Re-entrenar

- [ ] Tengo >200 imágenes por clase
- [ ] Las imágenes son de buena calidad
- [ ] Cada clase está balanceada (±20%)
- [ ] He revisado que no hay imágenes corruptas
- [ ] He hecho backup del modelo actual
- [ ] He ajustado hiperparámetros si es necesario
- [ ] Tengo espacio en disco para checkpoints

---

**💡 TIP FINAL**: La calidad de los datos es MÁS importante que la complejidad del modelo. Un modelo simple con 2,000 imágenes buenas supera a un modelo complejo con 200 imágenes malas.

¡Buena suerte mejorando el modelo! 🚀
