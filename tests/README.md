# 🧪 Tests del Sistema de Detección de Seguridad

Suite completa de pruebas para validar el funcionamiento del sistema.

## 📋 Tests Disponibles

### 1. **test_modelo.py** - Pruebas del Modelo CNN

Valida el modelo de detección con análisis regional y validación inteligente.

**Ejecutar**:

```bash
.venv\Scripts\activate
python tests/test_modelo.py
```

**Tests incluidos**:

- ✅ Carga correcta del modelo
- ✅ Detección en imágenes del dataset
- ✅ Validación de thresholds y filtros anti-falsos positivos

### 2. **test_logic.py** - Pruebas de Lógica de Seguridad

Verifica el análisis de riesgo y generación de alertas.

**Ejecutar**:

```bash
.venv\Scripts\activate
python tests/test_logic.py
```

**Tests incluidos**:

- ✅ Identificación de objetos sospechosos
- ✅ Cálculo de nivel de riesgo
- ✅ Análisis de escenarios de seguridad
- ✅ Generación de alertas
- ✅ Registro de eventos

## 🚀 Ejecutar Todos los Tests

```bash
# Windows
.venv\Scripts\activate
python tests/test_modelo.py
python tests/test_logic.py

# Linux/Mac
source .venv/bin/activate
python tests/test_modelo.py
python tests/test_logic.py
```

## 📊 Resultados Esperados

Todos los tests deberían pasar (✅ PASS) si el sistema está correctamente configurado:

```
🎯 Total: X/X tests pasaron
🎉 ¡Todos los tests pasaron!
```

## ⚠️ Requisitos

1. **Modelo entrenado**: `models/modelo_seguridad_v4.keras` debe existir
2. **Dataset**: Imágenes en `data/raw/{arma,gorro,mascara,persona}/`
3. **Dependencias**: Ejecutar `pip install -r requirements.txt`
4. **Entorno virtual**: Activar `.venv`

## 🔍 Solución de Problemas

### Error: "No se pudo cargar el modelo"

```bash
# Verificar que existe el modelo
dir models\modelo_seguridad_v4.keras  # Windows
ls models/modelo_seguridad_v4.keras   # Linux/Mac
```

### Error: "No se encontraron imágenes"

```bash
# Verificar estructura de datos
dir data\raw\arma\      # Windows
dir data\raw\persona\   # Windows
```

### Error: "ModuleNotFoundError"

```bash
# Asegurar que el entorno virtual está activado
.venv\Scripts\activate              # Windows
source .venv/bin/activate           # Linux/Mac

# Reinstalar dependencias
pip install -r requirements.txt
```

## 📝 Notas

- Los tests de **modelo** requieren imágenes en el dataset para validación completa
- Los tests de **lógica** son independientes y no requieren el modelo CNN
- Algunos tests pueden crear archivos temporales en `logs/`
- Los tests validan el sistema de **validación inteligente** con filtros anti-falsos positivos
