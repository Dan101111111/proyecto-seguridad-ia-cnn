"""
Script de Entrenamiento para Modelo de Seguridad v4
Entrena el modelo con Transfer Learning para máxima precisión
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
from datetime import datetime

print("="*70)
print("ENTRENAMIENTO MODELO DE SEGURIDAD V4")
print("Transfer Learning con MobileNetV2")
print("="*70)

# ==================== CONFIGURACIÓN ====================
# Ajusta estos parámetros según tu hardware y tiempo disponible
BATCH_SIZE = 32          # Reduce a 16 si tienes poca RAM
EPOCHS = 100             # Número máximo de épocas
IMG_SIZE = (224, 224)
DATA_DIR = 'data/raw'
LEARNING_RATE = 0.001    # Tasa de aprendizaje inicial

# ==================== CARGAR MODELO V4 ====================
print("\n📂 Cargando modelo v4 sin entrenar...")
try:
    model = keras.models.load_model('models/modelo_seguridad_v4.keras')
    print("✅ Modelo v4 cargado exitosamente!")
    print(f"\n📊 Arquitectura:")
    print(f"   - Capas totales: {len(model.layers)}")
    print(f"   - Parámetros entrenables: {model.count_params():,}")
    print(f"   - Transfer Learning: MobileNetV2")
except Exception as e:
    print(f"❌ ERROR: No se pudo cargar el modelo v4")
    print(f"   {e}")
    exit(1)

# ==================== PREPARAR DATOS ====================
print("\n📊 Preparando datos de entrenamiento...")

# Generador con Data Augmentation para mejorar generalización
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,        # Rotación hasta 30 grados
    width_shift_range=0.2,    # Desplazamiento horizontal
    height_shift_range=0.2,   # Desplazamiento vertical
    shear_range=0.2,          # Corte
    zoom_range=0.2,           # Zoom
    horizontal_flip=True,     # Volteo horizontal
    fill_mode='nearest',
    validation_split=0.2      # 80% train, 20% validation
)

# Cargar imágenes de entrenamiento
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Cargar imágenes de validación (sin augmentation)
validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"\n✅ Datos preparados:")
print(f"   Clases detectadas: {train_generator.class_indices}")
print(f"   Total imágenes entrenamiento: {train_generator.samples}")
print(f"   Total imágenes validación: {validation_generator.samples}")
print(f"   Batch size: {BATCH_SIZE}")

# Verificar que hay suficientes datos
if train_generator.samples < 50:
    print("\n⚠️  ADVERTENCIA: Pocas imágenes de entrenamiento")
    print("   Considera conseguir más datos para mejor accuracy")

# ==================== CONFIGURAR CALLBACKS ====================
print("\n⚙️  Configurando callbacks...")

# Crear directorio para checkpoints
os.makedirs('models/checkpoints', exist_ok=True)

# 1. ModelCheckpoint: Guarda el mejor modelo
checkpoint = ModelCheckpoint(
    'models/checkpoints/best_modelo_v4.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# 2. EarlyStopping: Detiene si no hay mejora
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=15,              # Espera 15 épocas sin mejora
    restore_best_weights=True,
    verbose=1
)

# 3. ReduceLROnPlateau: Reduce learning rate si se estanca
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,               # Reduce LR a la mitad
    patience=5,               # Después de 5 épocas sin mejora
    min_lr=1e-7,
    verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr]

# ==================== ENTRENAR ====================
print(f"\n🔥 Iniciando entrenamiento por hasta {EPOCHS} épocas...")
print("   (Se detendrá automáticamente si no hay mejora)")
print(f"   Tiempo estimado: ~{EPOCHS * train_generator.samples // (BATCH_SIZE * 60)} minutos")
print("\nPresiona Ctrl+C en cualquier momento para detener\n")
print("="*70)

try:
    # Iniciar entrenamiento
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*70)
    print("✅ Entrenamiento completado exitosamente!")
    
except KeyboardInterrupt:
    print("\n\n⚠️  Entrenamiento interrumpido por el usuario")
    print("Se guardarán los pesos del mejor modelo encontrado...")
    
except Exception as e:
    print(f"\n\n❌ ERROR durante el entrenamiento: {e}")
    exit(1)

# ==================== GUARDAR MODELO FINAL ====================
print("\n💾 Guardando modelos entrenados...")

try:
    # Guardar en ambos formatos
    model.save('models/modelo_seguridad_v4.h5')
    print("   ✅ modelo_seguridad_v4.h5 guardado")
    
    model.save('models/modelo_seguridad_v4.keras')
    print("   ✅ modelo_seguridad_v4.keras guardado")
    
    # Crear backup con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f'models/modelo_seguridad_v4_trained_{timestamp}.keras'
    model.save(backup_path)
    print(f"   ✅ Backup: {backup_path}")
    
except Exception as e:
    print(f"   ❌ Error al guardar: {e}")

# ==================== RESULTADOS ====================
print("\n" + "="*70)
print("📊 RESULTADOS DEL ENTRENAMIENTO")
print("="*70)

# Métricas finales
final_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print(f"\n📈 Métricas Finales:")
print(f"   Accuracy entrenamiento:  {final_acc*100:.2f}%")
print(f"   Accuracy validación:     {final_val_acc*100:.2f}%")
print(f"   Loss entrenamiento:      {final_loss:.4f}")
print(f"   Loss validación:         {final_val_loss:.4f}")

# Mejor época
best_epoch = history.history['val_accuracy'].index(max(history.history['val_accuracy'])) + 1
best_val_acc = max(history.history['val_accuracy'])
print(f"\n🏆 Mejor Época: {best_epoch}")
print(f"   Mejor Accuracy Validación: {best_val_acc*100:.2f}%")

# Evaluación del resultado
print("\n" + "="*70)
if final_val_acc > 0.90:
    print("🎉 ¡EXCELENTE! Modelo con accuracy >90%")
    print("   ✅ Listo para producción")
elif final_val_acc > 0.80:
    print("✅ ¡MUY BIEN! Modelo con accuracy >80%")
    print("   ✅ Aceptable para producción")
elif final_val_acc > 0.70:
    print("⚠️  ACEPTABLE. Modelo con accuracy >70%")
    print("   💡 Considera conseguir más datos o entrenar más épocas")
elif final_val_acc > 0.60:
    print("⚠️  REGULAR. Modelo con accuracy >60%")
    print("   💡 Necesita más datos o ajustar hiperparámetros")
else:
    print("❌ BAJO. Accuracy <60%")
    print("   💡 Revisa los datos o considera cambiar arquitectura")

# Análisis de overfitting
acc_diff = final_acc - final_val_acc
if acc_diff > 0.15:
    print("\n⚠️  ADVERTENCIA: Posible overfitting detectado")
    print(f"   Diferencia train-val: {acc_diff*100:.2f}%")
    print("   💡 Considera más data augmentation o regularización")
elif acc_diff < 0.05:
    print("\n✅ Excelente generalización del modelo")
else:
    print("\n✅ Buena generalización del modelo")

print("\n" + "="*70)
print("🎯 PRÓXIMOS PASOS")
print("="*70)
print("\n1. Probar el modelo entrenado:")
print("   python test_modelo_v4.py")
print("\n2. Probar en Streamlit UI:")
print("   streamlit run ui/app.py")
print("\n3. Si estás conforme, hacer commit:")
print("   git add models/modelo_seguridad_v4.*")
print("   git commit -m 'Modelo v4 entrenado con accuracy XX%'")
print("   git push origin daniel/ui-integration")

print("\n" + "="*70)
print(f"Entrenamiento finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)
