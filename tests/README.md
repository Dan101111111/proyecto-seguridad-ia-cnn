# Tests Directory

Carpeta para pruebas unitarias e integración del proyecto.

## Estructura recomendada:

```
tests/
├── test_detector.py        # Tests del módulo detector
├── test_preprocessing.py   # Tests de preprocesamiento
├── test_logic.py          # Tests de lógica de seguridad
└── test_utils.py          # Tests de utilidades
```

## Ejecutar tests:

```bash
# Instalar pytest
pip install pytest pytest-cov

# Ejecutar todos los tests
pytest tests/

# Con cobertura
pytest --cov=src tests/
```
