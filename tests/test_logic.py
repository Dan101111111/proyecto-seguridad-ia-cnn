"""
Test del módulo de lógica de seguridad
Verifica el funcionamiento de las funciones de análisis de riesgo
Autor: Bruno
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.logic import (
    check_security_risk,
    calculate_risk_level,
    is_suspicious_object,
    generate_alert,
    log_security_event,
    SUSPICIOUS_OBJECTS
)


def test_is_suspicious_object():
    """Test de identificación de objetos sospechosos"""
    print("\n--- TEST: is_suspicious_object ---")
    
    test_cases = [
        ('arma', True),
        ('weapon', True),
        ('gun', True),
        ('mascara', True),
        ('gorro', True),
        ('persona', False),
        ('car', False),
        ('ARMA', True),  # Case insensitive
    ]
    
    passed = 0
    failed = 0
    
    for label, expected in test_cases:
        result = is_suspicious_object(label)
        status = "✓" if result == expected else "✗"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
            
        print(f"  {status} '{label}' → {result} (esperado: {expected})")
    
    print(f"\nResultado: {passed} pasaron, {failed} fallaron")
    return failed == 0


def test_calculate_risk_level():
    """Test de cálculo de nivel de riesgo"""
    print("\n--- TEST: calculate_risk_level ---")
    
    test_cases = [
        ([], 'bajo', 'Sin objetos'),
        ([{'label': 'arma', 'confidence': 0.95}], 'crítico', 'Arma detectada'),
        ([{'label': 'gun', 'confidence': 0.88}], 'crítico', 'Gun detectado'),
        ([{'label': 'mascara', 'confidence': 0.85}], 'alto', 'Máscara alta confianza'),
        ([{'label': 'mascara', 'confidence': 0.65}], 'medio', 'Máscara baja confianza'),
        ([{'label': 'gorro', 'confidence': 0.70}], 'medio', 'Gorro detectado'),
    ]
    
    passed = 0
    failed = 0
    
    for objects, expected, description in test_cases:
        result = calculate_risk_level(objects)
        status = "✓" if result == expected else "✗"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
            
        print(f"  {status} {description}: {result} (esperado: {expected})")
    
    print(f"\nResultado: {passed} pasaron, {failed} fallaron")
    return failed == 0


def test_check_security_risk():
    """Test del análisis completo de riesgo"""
    print("\n--- TEST: check_security_risk ---")
    
    # Escenario 1: Normal (solo persona)
    print("\n  [Escenario 1] Normal - Solo persona")
    detections1 = [
        {'label': 'persona', 'confidence': 0.95}
    ]
    result1 = check_security_risk(detections1)
    
    assert result1['risk_level'] == 'bajo', "Esperaba nivel bajo"
    assert len(result1['suspicious_objects']) == 0, "No debería haber objetos sospechosos"
    assert not result1['alert_required'], "No debería requerir alerta"
    print(f"    ✓ Nivel: {result1['risk_level']}, Score: {result1['risk_score']:.2f}")
    
    # Escenario 2: Sospechoso (persona con gorro)
    print("\n  [Escenario 2] Sospechoso - Persona con gorro")
    detections2 = [
        {'label': 'persona', 'confidence': 0.88},
        {'label': 'gorro', 'confidence': 0.76}
    ]
    result2 = check_security_risk(detections2)
    
    assert result2['risk_level'] == 'medio', "Esperaba nivel medio"
    assert len(result2['suspicious_objects']) == 1, "Debería detectar 1 objeto sospechoso"
    assert result2['alert_required'], "Debería requerir alerta"
    print(f"    ✓ Nivel: {result2['risk_level']}, Score: {result2['risk_score']:.2f}")
    print(f"    ✓ Objetos sospechosos: {result2['suspicious_objects']}")
    
    # Escenario 3: Alto riesgo (máscara)
    print("\n  [Escenario 3] Alto riesgo - Persona con máscara")
    detections3 = [
        {'label': 'persona', 'confidence': 0.90},
        {'label': 'mascara', 'confidence': 0.85}
    ]
    result3 = check_security_risk(detections3)
    
    assert result3['risk_level'] == 'alto', "Esperaba nivel alto"
    assert result3['alert_required'], "Debería requerir alerta"
    print(f"    ✓ Nivel: {result3['risk_level']}, Score: {result3['risk_score']:.2f}")
    
    # Escenario 4: Crítico (arma)
    print("\n  [Escenario 4] Crítico - Arma detectada")
    detections4 = [
        {'label': 'arma', 'confidence': 0.92}
    ]
    result4 = check_security_risk(detections4)
    
    assert result4['risk_level'] == 'crítico', "Esperaba nivel crítico"
    assert result4['alert_required'], "Debería requerir alerta"
    print(f"    ✓ Nivel: {result4['risk_level']}, Score: {result4['risk_score']:.2f}")
    
    print("\n  ✓ Todos los escenarios pasaron")
    return True


def test_generate_alert():
    """Test de generación de alertas"""
    print("\n--- TEST: generate_alert ---")
    
    suspicious_objects = [
        {'label': 'arma', 'confidence': 0.92},
        {'label': 'mascara', 'confidence': 0.78}
    ]
    
    # Test para cada nivel
    levels = ['bajo', 'medio', 'alto', 'crítico']
    
    for level in levels:
        alert = generate_alert(level, suspicious_objects)
        
        assert isinstance(alert, str), "Alert debe ser un string"
        assert level.upper() in alert, f"Debe contener el nivel {level}"
        assert 'Timestamp' in alert, "Debe contener timestamp"
        
        print(f"\n  [Nivel: {level}]")
        print(f"  {alert[:80]}...")  # Mostrar solo primeras líneas
    
    print("\n  ✓ Todas las alertas generadas correctamente")
    return True


def test_log_security_event():
    """Test de registro de eventos"""
    print("\n--- TEST: log_security_event ---")
    
    # Crear evento de prueba
    event = {
        'risk_level': 'crítico',
        'location': 'Cámara Test',
        'detections': [
            {'label': 'arma', 'confidence': 0.95}
        ]
    }
    
    # Registrar evento
    result = log_security_event(event)
    
    assert result['success'], "Registro debe ser exitoso"
    assert 'message' in result, "Debe contener mensaje"
    
    print(f"  ✓ Evento registrado: {result['message']}")
    
    # Verificar que se creó el archivo
    log_file = 'logs/security_events.json'
    if os.path.exists(log_file):
        print(f"  ✓ Archivo de log creado en: {log_file}")
        
        # Leer y verificar contenido
        import json
        with open(log_file, 'r') as f:
            events = json.load(f)
        print(f"  ✓ Total de eventos en log: {len(events)}")
    
    return True


def test_integration_with_detector():
    """Test de integración simulando salida del detector de Igor"""
    print("\n--- TEST: Integración con detector.py ---")
    
    # Simular diferentes salidas del detector
    scenarios = [
        {
            'name': 'Normal',
            'detections': [{'label': 'persona', 'confidence': 0.95}],
            'expected_risk': 'bajo'
        },
        {
            'name': 'Gorro',
            'detections': [
                {'label': 'persona', 'confidence': 0.88},
                {'label': 'gorro', 'confidence': 0.76}
            ],
            'expected_risk': 'medio'
        },
        {
            'name': 'Máscara',
            'detections': [
                {'label': 'persona', 'confidence': 0.90},
                {'label': 'mascara', 'confidence': 0.85}
            ],
            'expected_risk': 'alto'
        },
        {
            'name': 'Arma',
            'detections': [{'label': 'arma', 'confidence': 0.92}],
            'expected_risk': 'crítico'
        }
    ]
    
    for scenario in scenarios:
        print(f"\n  [Escenario: {scenario['name']}]")
        
        # Analizar riesgo
        result = check_security_risk(scenario['detections'])
        
        # Verificar resultado esperado
        assert result['risk_level'] == scenario['expected_risk'], \
            f"Esperaba {scenario['expected_risk']}, obtuvo {result['risk_level']}"
        
        print(f"    ✓ Detecciones: {scenario['detections']}")
        print(f"    ✓ Riesgo: {result['risk_level']} (score: {result['risk_score']:.2f})")
        
        # Generar alerta si es necesario
        if result['alert_required']:
            alert = generate_alert(result['risk_level'], result['suspicious_objects'])
            print(f"    ✓ Alerta generada")
    
    print("\n  ✓ Integración completa verificada")
    return True


def run_all_tests():
    """Ejecutar todos los tests"""
    print("=" * 70)
    print("TEST DEL MÓDULO DE LÓGICA DE SEGURIDAD")
    print("Autor: Bruno")
    print("=" * 70)
    
    # Mostrar configuración
    print(f"\n[Configuración]")
    print(f"Objetos sospechosos definidos: {len(SUSPICIOUS_OBJECTS)}")
    print(f"Lista: {SUSPICIOUS_OBJECTS}")
    
    tests = [
        ("Identificación de objetos", test_is_suspicious_object),
        ("Cálculo de nivel de riesgo", test_calculate_risk_level),
        ("Análisis completo de seguridad", test_check_security_risk),
        ("Generación de alertas", test_generate_alert),
        ("Registro de eventos", test_log_security_event),
        ("Integración con detector", test_integration_with_detector),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n  ✗ ERROR: {e}")
            failed += 1
    
    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE TESTS")
    print("=" * 70)
    print(f"Total de tests: {len(tests)}")
    print(f"✓ Pasaron: {passed}")
    print(f"✗ Fallaron: {failed}")
    
    if failed == 0:
        print("\n🎉 ¡TODOS LOS TESTS PASARON!")
    else:
        print(f"\n⚠️  {failed} test(s) fallaron")
    
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
