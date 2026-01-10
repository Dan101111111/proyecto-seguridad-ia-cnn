# 🔒 Sistema de Detección de Seguridad con CNN

Aplicación de seguridad inteligente que detecta objetos sospechosos en tiempo real utilizando redes neuronales convolucionales (CNN).

## 📋 Descripción

Sistema de vigilancia automatizada que analiza imágenes y video para identificar objetos potencialmente peligrosos o sospechosos, generando alertas en tiempo real basadas en análisis de riesgo.

## 🚀 Características

- Detección de objetos en tiempo real usando CNN
- Análisis de riesgo de seguridad automatizado
- Interfaz web intuitiva con Streamlit
- Soporte para imágenes estáticas y video en vivo
- Registro de eventos de seguridad
- Sistema de alertas configurable

## 🛠️ Tecnologías

- **Python 3.8+**
- **TensorFlow/Keras** - Framework de Deep Learning
- **PyTorch** - Framework alternativo de DL
- **OpenCV** - Procesamiento de visión computacional
- **Streamlit** - Interfaz web interactiva
- **NumPy/Pandas** - Procesamiento de datos

## 📦 Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/proyecto-seguridad-ia-cnn.git
cd proyecto-seguridad-ia-cnn
```

2. Crear entorno virtual:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## 🎯 Uso

Ejecutar la aplicación:
```bash
streamlit run ui/app.py
```

La aplicación se abrirá en `http://localhost:8501`

## 📁 Estructura del Proyecto

```
├── data/              # Datasets y datos de entrenamiento
├── models/            # Modelos CNN entrenados
├── src/               # Código fuente principal
│   ├── detector.py    # Módulo de detección
│   ├── preprocessing.py  # Preprocesamiento de imágenes
│   ├── logic.py       # Lógica de seguridad
│   └── utils.py       # Utilidades generales
├── ui/                # Interfaz de usuario
│   ├── app.py         # Aplicación Streamlit
│   └── assets/        # Recursos estáticos
├── tests/             # Pruebas unitarias
├── requirements.txt   # Dependencias
└── README.md          # Documentación
```

## 🔧 Configuración

Ajustar parámetros en la barra lateral de la aplicación:
- Umbral de confianza de detección
- Nivel de riesgo de seguridad
- Modelo CNN a utilizar

## 📝 Licencia

Este proyecto es de código abierto.

## 👥 Autor

Daniell - [GitHub](https://github.com/tu-usuario)
