# 🚀 Guía de Deployment en Streamlit Cloud

## 📋 Pasos para Deploy

### 1. Preparar Repositorio en GitHub

Asegúrate de que tu repositorio tenga estos archivos:
- ✅ `app.py` (aplicación principal)
- ✅ `requirements.txt` (dependencias optimizadas)
- ✅ `packages.txt` (dependencias del sistema)
- ✅ `.streamlit/config.toml` (configuración de Streamlit)
- ✅ `models/modelo_seguridad_v4.keras` (modelo entrenado)

### 2. Subir a Streamlit Cloud

1. **Ir a**: https://share.streamlit.io/
2. **Iniciar sesión** con tu cuenta de GitHub
3. **Hacer clic** en "New app"
4. **Configurar**:
   - Repository: `Dan101111111/proyecto-seguridad-ia-cnn`
   - Branch: `main`
   - Main file path: `app.py`
5. **Click** en "Deploy!"

### 3. Esperar Deployment

El proceso puede tardar 5-10 minutos la primera vez:
- ⏳ Instalando dependencias...
- ⏳ Cargando modelo...
- ✅ ¡App desplegada!

## ⚙️ Configuración

### Archivos Clave

**`.streamlit/config.toml`**
- Tema oscuro configurado
- Tamaño máximo de upload: 200MB
- CORS deshabilitado para seguridad

**`requirements.txt`**
- TensorFlow CPU (más ligero que GPU para cloud)
- OpenCV headless (sin GUI)
- Versiones fijadas para estabilidad

**`packages.txt`**
- Dependencias del sistema Linux para OpenCV
- Necesarias para procesamiento de imágenes

## 🔧 Optimizaciones Aplicadas

1. **TensorFlow CPU** en vez de GPU (Streamlit Cloud no tiene GPU)
2. **opencv-python-headless** (más ligero, sin interfaz gráfica)
3. **Dependencias del sistema** para compatibilidad con Linux
4. **Límite de upload** de 200MB para videos grandes
5. **Tema oscuro** configurado por defecto

## 📊 Uso de Recursos

**Límites de Streamlit Cloud (Free Tier):**
- CPU: 1 core
- RAM: 1 GB
- Almacenamiento: 1 GB
- Ancho de banda: Ilimitado

**Recomendaciones:**
- El modelo v4.keras (~90MB) cabe perfectamente
- Videos hasta 200MB
- Imágenes sin límite práctico

## 🐛 Troubleshooting

### Error: "No module named 'cv2'"
**Solución**: Asegúrate que `packages.txt` existe con las dependencias del sistema

### Error: "Out of memory"
**Solución**: Reduce el tamaño del modelo o usa procesamiento por lotes

### Error: "Model not found"
**Solución**: Verifica que `models/modelo_seguridad_v4.keras` esté en el repo

### La app es muy lenta
**Solución**: 
- Usa cache de Streamlit (`@st.cache_resource`)
- Reduce FPS en webcam
- Procesa menos frames en video

## 🔒 Seguridad

- ✅ CORS deshabilitado
- ✅ XSRF protection habilitado
- ✅ No se recopilan estadísticas de uso
- ✅ Headless mode activado

## 📞 Soporte

Si tienes problemas con el deployment:
1. Verifica los logs en Streamlit Cloud
2. Revisa que todos los archivos estén en GitHub
3. Confirma que el modelo esté incluido en el repo

## 🎉 ¡Listo!

Una vez desplegado, tu app estará disponible en:
`https://[tu-app-name].streamlit.app`

Comparte el link con quien quieras probar el sistema de detección. 🚀
