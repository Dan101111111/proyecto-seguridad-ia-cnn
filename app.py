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
    
    if 'diagnostic_mode' not in st.session_state:
        st.session_state.diagnostic_mode = False
    
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False
    
    if 'confidence_threshold' not in st.session_state:
        # Threshold más bajo (0.3) para detectar múltiples objetos en seguridad
        st.session_state.confidence_threshold = 0.3
    
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
    
    # Modo diagnóstico
    st.session_state.diagnostic_mode = st.sidebar.checkbox(
        "Modo Diagnóstico",
        value=False,
        help="Muestra todas las probabilidades del modelo (incluso muy bajas)"
    )
    
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
                
                # Modo diagnóstico: mostrar todas las probabilidades
                if st.session_state.get('diagnostic_mode', False):
                    st.divider()
                    st.subheader("🔬 Diagnóstico del Modelo")
                    
                    # Obtener predicciones raw
                    input_frame = preprocess_frame(image_rgb)
                    input_frame_expanded = np.expand_dims(input_frame, axis=0)
                    raw_predictions = st.session_state.model.predict(input_frame_expanded, verbose=0)
                    
                    clases = ['arma', 'gorro', 'mascara', 'persona']
                    
                    st.markdown("**Probabilidades raw del modelo:**")
                    for idx, clase in enumerate(clases):
                        prob = raw_predictions[0][idx]
                        bar_color = "🔴" if clase == 'arma' and prob > 0.01 else "🟢"
                        st.write(f"{bar_color} **{clase}**: {prob:.4%} ({prob:.6f})")
                    
                    st.caption("💡 El modelo usa softmax, las probabilidades suman 100%")
                    st.caption("⚠️ Threshold base de arma: 8% (con validación regional: 1.5%)")
                    
                    # Mostrar estado de validación de arma
                    arma_detected = any(d['label'] == 'arma' for d in detections)
                    arma_prob = raw_predictions[0][0]
                    
                    if arma_prob > 0.01:
                        st.markdown("---")
                        st.markdown("**⚙️ Sistema de Validación de Armas:**")
                        
                        if arma_detected:
                            arma_det = next(d for d in detections if d['label'] == 'arma')
                            validation = arma_det.get('validation', {})
                            st.success(f"✅ Arma DETECTADA y VALIDADA")
                            st.write(f"   - Nivel de confianza: {validation.get('confidence_level', 'N/A')}")
                            st.write(f"   - Razón: {validation.get('reason', 'N/A')}")
                        else:
                            st.warning(f"⚠️ Arma detectada ({arma_prob:.2%}) pero FILTRADA como FALSO POSITIVO")
                            st.write("   - Razón: No cumplió criterios de validación")
                            st.write("   - El sistema protege contra falsos positivos")
                    
                    st.caption("🛡️ Sistema anti-falsos positivos activo")
                
        except Exception as e:
            st.error(f"❌ Error al procesar la imagen: {str(e)}")
            logger.error(f"Error en detección de imagen: {e}")


def video_detection_tab():
    """
    Tab para detección en video/webcam
    """
    st.header("🎥 Detección en Video")
    
    option = st.radio("Seleccionar fuente:", ["Subir video", "Usar webcam"])
    
    if option == "Subir video":
        st.subheader("📹 Cargar Video")
        uploaded_video = st.file_uploader(
            "Selecciona un video", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Formatos soportados: MP4, AVI, MOV, MKV"
        )
        
        if uploaded_video:
            # Configuración
            col1, col2 = st.columns(2)
            with col1:
                skip_frames = st.slider("Analizar cada N frames", 1, 30, 5, 
                                       help="Mayor valor = más rápido pero menos preciso")
            with col2:
                max_frames = st.slider("Frames máximos a procesar", 10, 500, 100,
                                      help="Limitar para videos largos")
            
            if st.button("🚀 Iniciar Análisis de Video", type="primary"):
                process_video(uploaded_video, skip_frames, max_frames)
    
    else:
        st.subheader("📷 Webcam en Tiempo Real")
        
        # Control de FPS
        fps_limit = st.slider("FPS objetivo", 1, 30, 10,
                             help="Controla la velocidad de análisis (mayor FPS = más fluido)")
        
        # Botones de control
        col1, col2 = st.columns(2)
        with col1:
            start_webcam = st.button("▶️ Iniciar Webcam", type="primary")
        with col2:
            stop_webcam = st.button("⏹️ Detener")
        
        if start_webcam:
            st.session_state.webcam_active = True
        if stop_webcam:
            st.session_state.webcam_active = False
        
        # Placeholder para video
        video_placeholder = st.empty()
        detection_placeholder = st.empty()
        
        if st.session_state.get('webcam_active', False):
            run_webcam_detection(video_placeholder, detection_placeholder, fps_limit)


def process_video(uploaded_video, skip_frames=5, max_frames=100):
    """
    Procesa un video subido y detecta objetos
    """
    import tempfile
    import os
    
    # Guardar video temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_video.read())
        video_path = tmp_file.name
    
    try:
        # Abrir video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        st.info(f"📹 Video: {total_frames} frames, {fps} FPS")
        
        # Contenedores
        progress_bar = st.progress(0)
        status_text = st.empty()
        video_col, stats_col = st.columns([2, 1])
        
        with video_col:
            frame_placeholder = st.empty()
        
        with stats_col:
            stats_placeholder = st.empty()
        
        # Procesar frames
        all_detections = []
        frame_count = 0
        processed_count = 0
        
        while cap.isOpened() and processed_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Solo procesar cada N frames
            if frame_count % skip_frames == 0:
                # Detectar objetos
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = detect_objects(
                    frame_rgb, 
                    st.session_state.model,
                    threshold=st.session_state.confidence_threshold
                )
                
                # Guardar detecciones
                if detections:
                    all_detections.append({
                        'frame': frame_count,
                        'time': frame_count / fps,
                        'detections': detections
                    })
                
                # Dibujar y mostrar
                frame_with_det = draw_detections(frame_rgb, detections)
                frame_placeholder.image(frame_with_det, channels="RGB", width=640)
                
                # Actualizar estadísticas
                update_video_stats(stats_placeholder, all_detections, processed_count, max_frames)
                
                processed_count += 1
            
            frame_count += 1
            progress_bar.progress(min(processed_count / max_frames, 1.0))
            status_text.text(f"Frame {frame_count}/{total_frames} (Procesados: {processed_count})")
        
        cap.release()
        
        # Resumen final
        st.success(f"✅ Análisis completado: {processed_count} frames procesados")
        show_video_summary(all_detections, fps)
        
    finally:
        # Limpiar archivo temporal
        if os.path.exists(video_path):
            os.unlink(video_path)


def run_webcam_detection(video_placeholder, detection_placeholder, fps_limit=10):
    """
    Ejecuta detección en webcam en tiempo real
    """
    import time
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ No se pudo acceder a la webcam")
        st.session_state.webcam_active = False
        return
    
    frame_time = 1.0 / fps_limit
    last_process_time = 0
    
    while st.session_state.get('webcam_active', False):
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()
        
        # Limitar FPS
        if current_time - last_process_time >= frame_time:
            # Convertir a RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detectar
            detections = detect_objects(
                frame_rgb,
                st.session_state.model,
                threshold=st.session_state.confidence_threshold
            )
            
            # Dibujar
            frame_with_det = draw_detections(frame_rgb, detections)
            
            # Mostrar frame
            video_placeholder.image(frame_with_det, channels="RGB", width=640)
            
            # Mostrar detecciones
            if detections:
                detection_placeholder.write(f"🔍 Detectados: {', '.join([d['label'] for d in detections])}")
            
            last_process_time = current_time
        
        time.sleep(0.01)  # Pequeña pausa para no saturar CPU
    
    cap.release()
    st.info("Webcam detenida")


def update_video_stats(placeholder, all_detections, processed, total):
    """
    Actualiza estadísticas del video en tiempo real
    """
    if not all_detections:
        placeholder.info("Sin detecciones aún...")
        return
    
    # Contar detecciones por clase
    class_counts = {}
    for record in all_detections:
        for det in record['detections']:
            label = det['label']
            class_counts[label] = class_counts.get(label, 0) + 1
    
    # Mostrar
    with placeholder.container():
        st.markdown("**📊 Estadísticas:**")
        st.metric("Frames analizados", processed)
        st.metric("Frames con detecciones", len(all_detections))
        
        if class_counts:
            st.markdown("**Detecciones:**")
            for label, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                st.write(f"- {label}: {count}")


def show_video_summary(all_detections, fps):
    """
    Muestra resumen final del análisis de video
    """
    st.divider()
    st.subheader("📊 Resumen del Análisis")
    
    if not all_detections:
        st.info("No se detectaron objetos sospechosos en el video")
        return
    
    # Métricas generales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Frames con detecciones", len(all_detections))
    
    with col2:
        times = [d['time'] for d in all_detections]
        st.metric("Primera detección", f"{min(times):.1f}s")
    
    with col3:
        st.metric("Última detección", f"{max(times):.1f}s")
    
    # Timeline de detecciones críticas
    critical_detections = [d for d in all_detections 
                          if any(det['label'] == 'arma' for det in d['detections'])]
    
    if critical_detections:
        st.error(f"🚨 {len(critical_detections)} frames con ARMAS detectadas")
        
        st.markdown("**⚠️ Timestamps críticos:**")
        for det_record in critical_detections[:10]:  # Mostrar primeros 10
            time_str = f"{int(det_record['time']//60):02d}:{int(det_record['time']%60):02d}"
            st.write(f"- Frame {det_record['frame']} ({time_str})")


def save_snapshot(frame, detections):
    """
    Guarda una captura de pantalla de la webcam
    """
    import os
    from datetime import datetime
    
    output_dir = "results/snapshots"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"snapshot_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    
    # Convertir RGB a BGR para guardar
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, frame_bgr)
    
    # Guardar también las detecciones en JSON
    import json
    json_path = filepath.replace('.jpg', '.json')
    
    # Convertir numpy types a tipos nativos de Python para JSON
    serializable_detections = []
    for det in detections:
        det_serializable = {}
        for key, value in det.items():
            # Convertir numpy/tensorflow tipos a Python nativos
            if hasattr(value, 'item'):  # numpy/tensorflow scalar
                det_serializable[key] = value.item()
            elif isinstance(value, dict):
                # Recursivamente convertir diccionarios anidados
                det_serializable[key] = {
                    k: (v.item() if hasattr(v, 'item') else v) 
                    for k, v in value.items()
                }
            else:
                det_serializable[key] = value
        serializable_detections.append(det_serializable)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'detections': serializable_detections
        }, f, indent=2, ensure_ascii=False)


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
            is_primary = det.get('is_primary', False)
            warning = det.get('warning', None)
            validation = det.get('validation', None)
            source = det.get('source', 'global')
            
            # Determinar si es sospechoso
            from src.logic import is_suspicious_object
            is_susp = is_suspicious_object(label)
            
            # Iconos
            if is_susp and label == 'arma':
                icon = "🔴"
            elif is_susp:
                icon = "⚠️"
            else:
                icon = "✓"
            
            # Mostrar con badge si es primario
            primary_badge = " **(Primario)**" if is_primary else ""
            conf_display = f"{conf:.1%}" if conf > 0.1 else f"{conf:.2%}"
            
            # Indicador de validación para armas
            validation_badge = ""
            if label == 'arma' and validation:
                confidence_icons = {
                    'muy_baja': '⚠️ Muy Baja',
                    'baja': '🟡 Baja',
                    'media': '🟠 Media',
                    'alta': '🔴 Alta'
                }
                validation_level = validation.get('confidence_level', 'media')
                validation_badge = f" - Validación: {confidence_icons.get(validation_level, '?')}"
            
            # Source badge
            source_badge = f" ({source})" if source != 'global' else ""
            
            st.write(f"{icon} **{label.upper()}**{primary_badge} - Confianza: {conf_display}{source_badge}{validation_badge}")
            
            # Mostrar razón de validación para armas
            if label == 'arma' and validation:
                reason = validation.get('reason', '')
                if reason:
                    st.caption(f"   💡 {reason}")
            
            # Mostrar advertencia si existe
            if warning and conf < 0.1:
                st.caption(f"   ⚡ {warning}")
    
    # Alerta si hay riesgo
    if risk_data.get('alert_required', False):
        st.error(f"🚨 **ALERTA DE SEGURIDAD**: Se detectaron {suspicious_count} objeto(s) sospechoso(s)")
    else:
        st.success("✅ No se detectaron amenazas de seguridad")


if __name__ == "__main__":
    main()
