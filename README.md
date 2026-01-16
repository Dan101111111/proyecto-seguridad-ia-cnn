# 🔒 Sistema de Detección de Seguridad - CNN

Sistema de detección de objetos sospechosos (armas, máscaras, gorros) usando Deep Learning con validación inteligente anti-falsos positivos.

## 🌐 Demo en Línea

🔗 **Prueba la app en vivo**: [Próximamente en Streamlit Cloud]

## 🚀 Inicio Rápido

### Opción 1: Ejecutar Localmente

```bash
# 1. Activar entorno virtual
.venv\Scripts\activate

# 2. Ejecutar aplicación
streamlit run app.py
```

La aplicación se abrirá automáticamente en: **http://localhost:8501**

### Opción 2: Desplegar en Streamlit Cloud

Ver instrucciones completas en [DEPLOYMENT.md](DEPLOYMENT.md)

## 📋 Requisitos

- Python 3.8+
- Dependencias en `requirements.txt`

### Instalación

```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

## ✨ Características

### 🎯 Validación Inteligente

Sistema con 4 filtros anti-falsos positivos:

- ✅ Detecta armas reales correctamente
- ✅ NO genera falsos positivos en gestos de manos
- ✅ Análisis regional para detectar armas en manos
- ✅ Balance óptimo entre sensibilidad y precisión

### 📸 Modos de Detección

1. **Imagen**: Sube imágenes para análisis
2. **Video**: Procesa archivos de video
3. **Webcam**: Detección en tiempo real (hasta 30 FPS)

### 🔍 Clases Detectables

- Arma (pistolas, rifles, escopetas)
- Gorro
- Máscara
- Persona

## ⚙️ Configuración

### Thresholds Actuales

```python
Threshold arma global:    15%  # Sensible para pistolas
Threshold arma regional:  25%  # Balance detección/FP
Análisis regional:        ✓ ACTIVADO (3 regiones)
Validación inteligente:   ✓ ACTIVADA (4 filtros)
```

### Ajustar Sensibilidad

Para modificar la sensibilidad, edita [src/detector.py](src/detector.py) línea ~51:

**Más sensible** (detecta más):

```python
'arma': 0.12,  # Bajar threshold
```

**Más estricto** (menos falsos positivos):

```python
'arma': 0.20,  # Subir threshold
```

## 📁 Estructura del Proyecto

```
proyecto-seguridad-ia-cnn/
│
├── app.py                          # Aplicación principal Streamlit
├── requirements.txt                # Dependencias
├── config.json                     # Configuración
│
├── src/                            # Código fuente
│   ├── detector.py                 # Motor de detección con validación
│   ├── preprocessing.py            # Preprocesamiento de imágenes
│   ├── logic.py                    # Lógica de análisis de riesgo
│   └── utils.py                    # Utilidades
│
├── models/                         # Modelos entrenados
│   └── modelo_seguridad_v4.keras   # CNN (93% accuracy)
│
└── data/                           # Datasets
    └── raw/                        # Imágenes por clase
```

## 🧪 Pruebas

El proyecto incluye una suite completa de tests para validar el funcionamiento:

```bash
# Activar entorno virtual
.venv\Scripts\activate

# Probar el modelo CNN
python tests/test_modelo.py

# Probar la lógica de seguridad
python tests/test_logic.py
```

Ver más detalles en [tests/README.md](tests/README.md)

## 🎓 Modelo

**Versión**: v4.keras
**Arquitectura**: Transfer Learning con MobileNetV2
**Precisión**: 93.26% validación, 91.67% test
**Dataset**: 14,696 imágenes (arma, gorro, máscara, persona)

### Nota sobre Pistolas

El modelo fue entrenado principalmente con rifles y escopetas. Para pistolas pequeñas:

- **Actual**: Threshold optimizado a 15%
- **Futuro**: Re-entrenar con más imágenes de pistolas

## 🛡️ Sistema de Validación

Filtros anti-falsos positivos:

1. **Filtro 1**: Global < 5% + 1 región → RECHAZAR
2. **Filtro 2**: Global < 5% + 2 regiones débiles → RECHAZAR
3. **Filtro 3**: Global < 10% + 1 región débil → RECHAZAR
4. **Filtro 4**: Global ≥ 10% O múltiples regiones → VALIDAR

Esto permite detectar personas armadas (guardias, agentes) sin generar alertas falsas por gestos de manos.

## 📞 Soporte

Para problemas o preguntas, consulta la documentación en el código o contacta al equipo de desarrollo.
