"""
Aplicación Streamlit para detección de objetos sospechosos
Autor: Daniel - Líder de Integración y UI
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import logging
from pathlib import Path

# Importaciones desde src
from src.detector import detect_objects, load_model, draw_detections, get_detection_summary
from src.preprocessing import preprocess_frame, preprocess_image_for_display
from src.logic import check_security_risk, calculate_risk_level
from src.utils import save_image, get_timestamp, load_config, create_output_directory

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar configuración
try:
    CONFIG = load_config('config.json')
except:
    CONFIG = {
        'model': {'path': 'models/modelo_seguridad_v1.h5', 'threshold': 0.6},
        'security': {'risk_threshold': 0.7},
        'ui': {'save_detections': True, 'output_dir': 'results/detections'}
    }


def init_session_state():
    """
    Inicializa el estado de la sesión de Streamlit
    """
    if 'model' not in st.session_state:
        st.session_state.model = None
        st.session_state.model_loaded = False
    
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    
    if 'confidence_threshold' not in st.session_state:
        st.session_state.confidence_threshold = CONFIG.get('model', {}).get('threshold', 0.6)
    
    if 'risk_threshold' not in st.session_state:
        st.session_state.risk_threshold = CONFIG.get('security', {}).get('risk_threshold', 0.7)


@st.cache_resource
def get_model():
    """
    Carga y cachea el modelo CNN para evitar recargarlo en cada interacción
    """
    try:
        model_path = CONFIG.get('model', {}).get('path', 'models/modelo_seguridad_v1.h5')
        logger.info(f"Cargando modelo desde: {model_path}")
        model = load_model(model_path)
        if model:
            logger.info("Modelo cargado exitosamente")
        return model
    except Exception as e:
        logger.error(f"Error al cargar modelo: {e}")
        return None


def main():
    """
    Función principal de la aplicación Streamlit
    """
    st.set_page_config(
        page_title="Sistema de Detección de Seguridad",
        page_icon="🔒",
        layout="wide"
    )
    
    # Inicializar estado de sesión
    init_session_state()
    
    st.title("🔒 Sistema de Detección de Objetos Sospechosos")
    st.markdown("### Detección en tiempo real usando CNN")
    
    # Cargar modelo
    if not st.session_state.model_loaded:
        with st.spinner('Cargando modelo CNN...'):
            st.session_state.model = get_model()
            st.session_state.model_loaded = True
    
    # Verificar que el modelo se cargó
    if st.session_state.model is None:
        st.error("⚠️ No se pudo cargar el modelo. Verifica que el archivo exista en 'models/modelo_seguridad_v1.h5'")
        return
    
    # Sidebar con opciones
    setup_sidebar()
    
    # Tabs principales
    tab1, tab2, tab3 = st.tabs(["📸 Detección en Imagen", "🎥 Detección en Video", "📊 Historial"])
    
    with tab1:
        image_detection_tab()
    
    with tab2:
        video_detection_tab()
    
    with tab3:
        history_tab()


def setup_sidebar():
    """
    Configura el panel lateral con opciones
    """
    st.sidebar.header("⚙️ Configuración")
    
    # Actualizar umbrales en session_state
    st.session_state.confidence_threshold = st.sidebar.slider(
        "Umbral de Confianza", 
        0.0, 1.0, 
        st.session_state.confidence_threshold, 
        0.05,
        help="Nivel mínimo de confianza para considerar una detección válida"
    )
    
    st.session_state.risk_threshold = st.sidebar.slider(
        "Umbral de Riesgo", 
        0.0, 1.0, 
        st.session_state.risk_threshold, 
        0.05,
        help="Nivel mínimo de riesgo para generar una alerta"
    )
    
    st.sidebar.divider()
    
    # Información del modelo
    st.sidebar.subheader("📊 Info del Modelo")
    st.sidebar.info(f"**Clases detectables:**\n- Arma\n- Gorro\n- Máscara\n- Persona")
    
    st.sidebar.divider()
    
    # Opciones adicionales
    save_detections = st.sidebar.checkbox("Guardar Detecciones", value=True)
    
    if save_detections:
        st.sidebar.caption(f"📁 Guardando en: {CONFIG.get('ui', {}).get('output_dir', 'results/detections')}")
    
    return save_detections


def image_detection_tab():
    """
    Tab para detección en imágenes estáticas
    """
    st.header("📸 Detección en Imagen")
    st.markdown("Sube una imagen para detectar objetos sospechosos")
    
    uploaded_file = st.file_uploader(
        "Selecciona una imagen", 
        type=['jpg', 'jpeg', 'png'],
        help="Formatos soportados: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        try:
            # Leer imagen
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Mostrar imagen original y resultados
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Imagen Original")
                st.image(image_rgb, use_container_width=True)
            
            with col2:
                st.subheader("Resultados de Detección")
                
                with st.spinner('Analizando imagen...'):
                    # Realizar detección
                    detections = detect_objects(
                        image_rgb, 
                        st.session_state.model, 
                        threshold=st.session_state.confidence_threshold
                    )
                    
                    if detections:
                        # Dibujar detecciones
                        image_with_detections = draw_detections(image_rgb, detections)
                        st.image(image_with_detections, use_container_width=True)
                        
                        # Guardar en historial
                        st.session_state.detection_history.append({
                            'timestamp': get_timestamp(),
                            'detections': detections,
                            'image_name': uploaded_file.name
                        })
                    else:
                        st.info("✅ No se detectaron objetos sospechosos")
                        st.image(image_rgb, use_container_width=True)
            
            # Mostrar análisis de riesgo
            if detections:
                st.divider()
                show_risk_analysis(detections)
                
        except Exception as e:
            st.error(f"❌ Error al procesar la imagen: {str(e)}")
            logger.error(f"Error en detección de imagen: {e}")


def video_detection_tab():
    """
    Tab para detección en video/webcam
    """
    st.header("🎥 Detección en Video")
    st.info("📹 Esta funcionalidad estará disponible próximamente")
    
    option = st.radio("Seleccionar fuente:", ["Subir video", "Usar webcam"])
    
    if option == "Subir video":
        uploaded_video = st.file_uploader("Cargar video", type=['mp4', 'avi', 'mov'])
        if uploaded_video:
            st.warning("Procesamiento de video en desarrollo...")
    else:
        if st.button("Iniciar Webcam"):
            st.warning("Detección por webcam en desarrollo...")


def history_tab():
    """
    Tab para mostrar historial de detecciones
    """
    st.header("📊 Historial de Detecciones")
    
    if not st.session_state.detection_history:
        st.info("📋 No hay detecciones registradas aún. Sube una imagen para comenzar.")
        return
    
    st.markdown(f"**Total de detecciones:** {len(st.session_state.detection_history)}")
    
    # Mostrar historial en orden inverso (más reciente primero)
    for i, record in enumerate(reversed(st.session_state.detection_history), 1):
        with st.expander(f"🔍 Detección #{len(st.session_state.detection_history) - i + 1} - {record.get('image_name', 'Sin nombre')}"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write(f"**Fecha:** {record.get('timestamp', 'N/A')}")
                st.write(f"**Objetos detectados:** {len(record.get('detections', []))}")
            
            with col2:
                for det in record.get('detections', []):
                    st.write(f"- {det.get('label', 'N/A')}: {det.get('confidence', 0):.1%}")
    
    # Botón para limpiar historial
    if st.button("🗑️ Limpiar Historial"):
        st.session_state.detection_history = []
        st.rerun()


def show_risk_analysis(detections):
    """
    Muestra el análisis de riesgo de seguridad
    
    Args:
        detections: Lista de detecciones del modelo
    """
    st.subheader("📊 Análisis de Riesgo de Seguridad")
    
    # Análisis de riesgo usando la lógica de Bruno
    risk_data = check_security_risk(detections, st.session_state.risk_threshold)
    
    # Métricas principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_level = risk_data.get('risk_level', 'bajo')
        risk_colors = {
            'bajo': '🟢',
            'medio': '🟡',
            'alto': '🟠',
            'crítico': '🔴'
        }
        st.metric(
            "Nivel de Riesgo", 
            f"{risk_colors.get(risk_level, '⚪')} {risk_level.upper()}"
        )
    
    with col2:
        st.metric("Objetos Detectados", len(detections))
    
    with col3:
        suspicious_count = len(risk_data.get('suspicious_objects', []))
        st.metric("Objetos Sospechosos", suspicious_count)
    
    # Detalles de objetos detectados
    if detections:
        st.markdown("##### Detecciones:")
        for i, det in enumerate(detections, 1):
            label = det.get('label', 'Desconocido')
            conf = det.get('confidence', 0.0)
            is_suspicious = label in CONFIG.get('security', {}).get('suspicious_objects', [])
            
            icon = "⚠️" if is_suspicious else "✓"
            st.write(f"{icon} **{label}** - Confianza: {conf:.1%}")
    
    # Alerta si hay riesgo
    if risk_data.get('alert_required', False):
        st.error(f"🚨 **ALERTA DE SEGURIDAD**: Se detectaron {suspicious_count} objeto(s) sospechoso(s)")
    else:
        st.success("✅ No se detectaron amenazas de seguridad")


if __name__ == "__main__":
    main()
