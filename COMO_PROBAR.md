# 🧪 Cómo Probar el Modelo de Seguridad v2

Esta guía te muestra cómo probar el sistema completo de detección de seguridad.

---

## 🚀 Inicio Rápido

### 1. Activar entorno virtual

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate
```

### 2. Opción A: Probar con Script Automático

```bash
python test_modelo_v2.py
```

**Esto hace:**
- ✅ Carga modelo v2 (.h5 y .keras)
- ✅ Prueba con 3 imágenes de cada clase
- ✅ Calcula accuracy
- ✅ Muestra distribución de probabilidades

**Resultado esperado:**
```
============================================================
CONCLUSIÓN FINAL
============================================================
✅ Ambos modelos funcionan correctamente!
💡 Recomendación: Usar modelo_seguridad_v2.keras (formato más reciente)

📝 Próximos pasos:
   1. Actualizar config.json para usar modelo v2
   2. Probar en la interfaz Streamlit
   3. Hacer commit y push del nuevo modelo
```

---

### 2. Opción B: Probar con Interfaz Streamlit (RECOMENDADO)

```bash
streamlit run ui/app.py
```

**Se abrirá automáticamente:** http://localhost:8502

#### 📸 Prueba de Detección de Imágenes:

1. Ve a la pestaña **"🖼️ Detección en Imagen"**
2. Haz clic en **"Browse files"**
3. Sube una imagen de prueba (ej: `data/raw/arma/arma_001.jpg`)
4. Verás:
   - ✅ Imagen original
   - ✅ Imagen procesada (con detecciones)
   - ✅ Resultados de detección (clase + confianza)
   - ✅ Análisis de riesgo de seguridad

**Ejemplo de resultado:**

```
🎯 Resultados de Detección
─────────────────────────

📦 Objeto: arma
🎲 Confianza: 85.23%

⚠️ Análisis de Riesgo
─────────────────────────

🔴 NIVEL DE RIESGO: ALTO
Nivel de amenaza detectado: 0.85
```

#### 📜 Historial de Detecciones:

1. Ve a la pestaña **"📜 Historial"**
2. Verás todas las detecciones previas
3. Expande cualquier detección para ver detalles

---

## 🧪 Pruebas Recomendadas

### Test 1: Detectar Arma

```bash
# Desde PowerShell
streamlit run ui/app.py
```

1. Subir: `data/raw/arma/arma_001.jpg`
2. **Esperado:** Detecta "arma" con confianza > 60%
3. **Riesgo:** ALTO

### Test 2: Detectar Gorro/Casco

1. Subir: `data/raw/gorro/casco_001.jpg`
2. **Esperado:** Detecta "gorro" con confianza > 60%
3. **Riesgo:** MEDIO

### Test 3: Detectar Máscara

1. Subir: `data/raw/mascara/mask_001.jpg`
2. **Esperado:** Detecta "mascara" con confianza > 60%
3. **Riesgo:** MEDIO-ALTO

### Test 4: Detectar Persona

1. Subir: `data/raw/persona/persona_001.jpg`
2. **Esperado:** Detecta "persona" con confianza > 60%
3. **Riesgo:** BAJO

---

## 📊 Métricas de Éxito

### Modelo SIN entrenar (actual):
- ⚠️ Accuracy ~25% (predicción aleatoria)
- ⚠️ Confianza ~25% en todas las clases
- ⚠️ No diferencia entre clases

### Modelo ENTRENADO (después de que Igor entrene):
- ✅ Accuracy > 60% (mínimo aceptable)
- ✅ Accuracy > 80% (ideal)
- ✅ Confianza > 70% en clase correcta

---

## 🐛 Troubleshooting

### Error: "No module named 'tensorflow'"

```bash
# Verificar que el entorno virtual está activado
.venv\Scripts\Activate.ps1

# Reinstalar dependencias
pip install -r requirements.txt
```

### Error: "No module named 'src'"

```bash
# Asegurarse de ejecutar desde la raíz del proyecto
cd C:\Users\Daniel\Downloads\proyecto-seguridad-ia-cnn
$env:PYTHONPATH="C:\Users\Daniel\Downloads\proyecto-seguridad-ia-cnn"
streamlit run ui/app.py
```

### Warning: "albumentations no disponible"

- ⚠️ **Normal:** No afecta el funcionamiento básico
- 📝 **Info:** Albumentations es opcional (data augmentation avanzado)
- 🔧 **Solución (opcional):** `pip install albumentations`

### Modelo tiene baja accuracy (~25%)

- ⚠️ **Normal:** El modelo actual NO está entrenado
- 📝 **Solución:** Igor necesita entrenar el modelo
- 📖 **Ver:** [GUIA_MODELO_V2.md](GUIA_MODELO_V2.md)

---

## 📁 Estructura de Datos de Prueba

```
data/raw/
├── arma/        (60 imágenes)  → Riesgo ALTO
├── gorro/       (30 imágenes)  → Riesgo MEDIO
├── mascara/     (30 imágenes)  → Riesgo MEDIO-ALTO
└── persona/     (60 imágenes)  → Riesgo BAJO
```

---

## ✅ Checklist de Prueba Completa

Antes de dar el visto bueno al proyecto:

- [ ] `test_modelo_v2.py` ejecuta sin errores
- [ ] Streamlit UI carga correctamente
- [ ] Puedo subir una imagen y ver resultados
- [ ] Los resultados muestran clase + confianza
- [ ] El análisis de riesgo funciona (Bruno's module)
- [ ] El historial guarda las detecciones
- [ ] Puedo probar con las 4 clases diferentes

### Prueba Adicional (cuando modelo esté entrenado):

- [ ] Accuracy en test set > 60%
- [ ] Modelo detecta correctamente armas
- [ ] Modelo detecta correctamente gorros/cascos
- [ ] Modelo detecta correctamente máscaras
- [ ] Modelo detecta correctamente personas
- [ ] UI muestra nivel de riesgo apropiado

---

## 📞 Ayuda

**Si tienes problemas:**

1. Leer [GUIA_MODELO_V2.md](GUIA_MODELO_V2.md)
2. Revisar [PROBLEMA_MODELO_V2_IGOR.md](PROBLEMA_MODELO_V2_IGOR.md)
3. Contactar a:
   - **Daniel** - UI e integración
   - **Igor** - Modelo CNN
   - **D'Alessandro** - Preprocesamiento
   - **Bruno** - Lógica de riesgo

---

**Actualizado:** 15/01/2026  
**Estado:** ✅ Sistema funcional, pendiente entrenamiento del modelo
