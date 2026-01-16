"""
Test del módulo de lógica de seguridad
Verifica el funcionamiento de las funciones de análisis de riesgo
"""
import sys
import os
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    print("\n" + "="*60)
    print("TEST 1: IDENTIFICACIÓN DE OBJETOS SOSPECHOSOS")
    print("="*60)
    
    test_cases = [
        ('arma', True),
        ('weapon', True),
        ('gun', True),
        ('mascara', True),
        ('gorro', True),
        ('persona', False),
        ('car', False),
        ('ARMA', True),  # Case insensitive
        ('mask', True),
        ('hat', True),
    ]
    
    passed = 0
    failed = 0
    
    print("\nProbando detección de objetos sospechosos:\n")
    
    for label, expected in test_cases:
        result = is_suspicious_object(label)
        status = "✅" if result == expected else "❌"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
            
        print(f"  {status} '{label:10s}' → {'SOSPECHOSO' if result else 'NORMAL':12s} (esperado: {'SOSPECHOSO' if expected else 'NORMAL'})")
    
    print(f"\n📊 Resultado: {passed} pasaron, {failed} fallaron")
    return failed == 0


def test_calculate_risk_level():
    """Test de cálculo de nivel de riesgo"""
    print("\n" + "="*60)
    print("TEST 2: CÁLCULO DE NIVEL DE RIESGO")
    print("="*60)
    
    test_cases = [
        ([], 'bajo', 'Sin objetos detectados'),
        ([{'label': 'persona', 'confidence': 0.95}], 'bajo', 'Solo persona'),
        ([{'label': 'arma', 'confidence': 0.95}], 'crítico', 'Arma detectada'),
        ([{'label': 'gun', 'confidence': 0.88}], 'crítico', 'Gun detectado'),
        ([{'label': 'mascara', 'confidence': 0.85}], 'alto', 'Máscara alta confianza'),
        ([{'label': 'mascara', 'confidence': 0.65}], 'medio', 'Máscara baja confianza'),
        ([{'label': 'gorro', 'confidence': 0.70}], 'medio', 'Gorro detectado'),
        ([{'label': 'persona', 'confidence': 0.90}, {'label': 'gorro', 'confidence': 0.75}], 'medio', 'Persona con gorro'),
        ([{'label': 'persona', 'confidence': 0.88}, {'label': 'mascara', 'confidence': 0.82}], 'alto', 'Persona con máscara'),
    ]
    
    passed = 0
    failed = 0
    
    print("\nProbando cálculo de nivel de riesgo:\n")
    
    for objects, expected, description in test_cases:
        result = calculate_risk_level(objects)
        status = "✅" if result == expected else "❌"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
        
        # Determinar emoji según nivel
        emoji = {
            'bajo': '🟢',
            'medio': '🟡',
            'alto': '🟠',
            'crítico': '🔴'
        }.get(result, '⚪')
        
        print(f"  {status} {emoji} {description:30s} → {result:8s} (esperado: {expected})")
    
    print(f"\n📊 Resultado: {passed} pasaron, {failed} fallaron")
    return failed == 0


def test_check_security_risk():
    """Test del análisis completo de riesgo"""
    print("\n" + "="*60)
    print("TEST 3: ANÁLISIS DE RIESGO DE SEGURIDAD")
    print("="*60)
    
    scenarios = [
        {
            'name': 'Normal - Solo persona',
            'detections': [{'label': 'persona', 'confidence': 0.95}],
            'expected_risk': 'bajo',
            'expected_alert': False
        },
        {
            'name': 'Sospechoso - Persona con gorro',
            'detections': [
                {'label': 'persona', 'confidence': 0.88},
                {'label': 'gorro', 'confidence': 0.76}
            ],
            'expected_risk': 'medio',
            'expected_alert': True
        },
        {
            'name': 'Alto riesgo - Persona con máscara',
            'detections': [
                {'label': 'persona', 'confidence': 0.90},
                {'label': 'mascara', 'confidence': 0.85}
            ],
            'expected_risk': 'alto',
            'expected_alert': True
        },
        {
            'name': 'Crítico - Arma detectada',
            'detections': [{'label': 'arma', 'confidence': 0.92}],
            'expected_risk': 'crítico',
            'expected_alert': True
        },
    ]
    
    passed = 0
    failed = 0
    
    print("\nProbando escenarios de seguridad:\n")
    
    for scenario in scenarios:
        result = check_security_risk(scenario['detections'])
        
        risk_ok = result['risk_level'] == scenario['expected_risk']
        alert_ok = result['alert_required'] == scenario['expected_alert']
        suspicious_ok = len(result['suspicious_objects']) > 0 if scenario['expected_alert'] else True
        
        all_ok = risk_ok and alert_ok and suspicious_ok
        status = "✅" if all_ok else "❌"
        
        if all_ok:
            passed += 1
        else:
            failed += 1
        
        # Emoji según nivel
        emoji = {
            'bajo': '🟢',
            'medio': '🟡',
            'alto': '🟠',
            'crítico': '🔴'
        }.get(result['risk_level'], '⚪')
        
        print(f"  {status} {emoji} {scenario['name']:35s}")
        print(f"      Nivel: {result['risk_level']:8s} (esperado: {scenario['expected_risk']})")
        print(f"      Score: {result['risk_score']:.2f}")
        if result['suspicious_objects']:
            print(f"      Objetos: {[obj['label'] for obj in result['suspicious_objects']]}")
    
    print(f"\n📊 Resultado: {passed} pasaron, {failed} fallaron")
    return failed == 0


def test_generate_alert():
    """Test de generación de alertas"""
    print("\n" + "="*60)
    print("TEST 4: GENERACIÓN DE ALERTAS")
    print("="*60)
    
    suspicious_objects = [
        {'label': 'arma', 'confidence': 0.92},
        {'label': 'mascara', 'confidence': 0.78}
    ]
    
    levels = ['bajo', 'medio', 'alto', 'crítico']
    
    print("\nProbando generación de alertas:\n")
    
    passed = 0
    failed = 0
    
    for level in levels:
        try:
            alert = generate_alert(level, suspicious_objects)
            
            has_level = level.upper() in alert
            has_timestamp = 'Timestamp' in alert or 'TIMESTAMP' in alert
            is_string = isinstance(alert, str)
            
            all_ok = has_level and has_timestamp and is_string
            status = "✅" if all_ok else "❌"
            
            if all_ok:
                passed += 1
            else:
                failed += 1
            
            emoji = {
                'bajo': '🟢',
                'medio': '🟡',
                'alto': '🟠',
                'crítico': '🔴'
            }.get(level, '⚪')
            
            print(f"  {status} {emoji} Alerta nivel '{level}' generada")
            
            # Mostrar primeras líneas
            lines = alert.split('\n')[:3]
            for line in lines:
                if line.strip():
                    print(f"      {line[:70]}")
            
        except Exception as e:
            print(f"  ❌ Error generando alerta '{level}': {e}")
            failed += 1
    
    print(f"\n📊 Resultado: {passed} pasaron, {failed} fallaron")
    return failed == 0


def test_log_security_event():
    """Test de registro de eventos"""
    print("\n" + "="*60)
    print("TEST 5: REGISTRO DE EVENTOS DE SEGURIDAD")
    print("="*60)
    
    # Crear evento de prueba
    event = {
        'risk_level': 'crítico',
        'location': 'Cámara Test',
        'detections': [
            {'label': 'arma', 'confidence': 0.95}
        ]
    }
    
    print("\nProbando registro de eventos:\n")
    
    try:
        # Registrar evento
        result = log_security_event(event)
        
        if result.get('success'):
            print(f"  ✅ Evento registrado exitosamente")
            print(f"      Mensaje: {result.get('message', 'N/A')}")
            
            # Verificar que se creó el archivo
            log_file = 'logs/security_events.json'
            if os.path.exists(log_file):
                print(f"  ✅ Archivo de log creado: {log_file}")
                
                # Leer y verificar contenido
                import json
                try:
                    with open(log_file, 'r') as f:
                        events = json.load(f)
                    print(f"  ✅ Total de eventos en log: {len(events)}")
                except:
                    print(f"  ⚠️  Archivo creado pero no se pudo leer el contenido")
            
            return True
        else:
            print(f"  ❌ Error: {result.get('message', 'Desconocido')}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error al registrar evento: {e}")
        return False


def run_all_tests():
    """Ejecuta todos los tests de lógica"""
    print("\n" + "="*60)
    print("🧪 SUITE DE PRUEBAS - MÓDULO DE LÓGICA")
    print("="*60)
    
    results = []
    
    # Ejecutar todos los tests
    results.append(("Identificación de objetos sospechosos", test_is_suspicious_object()))
    results.append(("Cálculo de nivel de riesgo", test_calculate_risk_level()))
    results.append(("Análisis de riesgo de seguridad", test_check_security_risk()))
    results.append(("Generación de alertas", test_generate_alert()))
    results.append(("Registro de eventos", test_log_security_event()))
    
    # Resumen final
    print("\n" + "="*60)
    print("📊 RESUMEN FINAL")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\n🎯 Total: {total_passed}/{total_tests} tests pasaron")
    
    if total_passed == total_tests:
        print("🎉 ¡Todos los tests pasaron!")
        return True
    else:
        print(f"⚠️  {total_tests - total_passed} test(s) fallaron")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
