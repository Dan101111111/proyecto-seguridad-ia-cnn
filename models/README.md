# Models Directory

Esta carpeta almacena los modelos CNN entrenados.

## Formatos soportados:
- `.h5` - Modelos de Keras/TensorFlow
- `.pth` / `.pt` - Modelos de PyTorch
- `.onnx` - Formato ONNX para portabilidad
- `.pb` - TensorFlow SavedModel

## Notas:
- Los archivos de modelos se ignoran en .gitignore por su tamaño
- Usar versionado externo (DVC, MLflow) para modelos en producción
- Guardar aquí checkpoints y modelos finales
