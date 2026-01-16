"""
Script de verificación pre-deployment para Streamlit Cloud
Verifica que todos los archivos necesarios estén presentes
"""
import os
import sys
from pathlib import Path

def check_file(filepath, required=True):
    """Verifica si un archivo existe"""
    exists = Path(filepath).exists()
    status = "✅" if exists else ("❌" if required else "⚠️")
    req_text = "REQUERIDO" if required else "OPCIONAL"
    print(f"{status} {filepath} - {req_text}")
    return exists

def check_directory(dirpath, required=True):
    """Verifica si un directorio existe"""
    exists = Path(dirpath).exists() and Path(dirpath).is_dir()
    status = "✅" if exists else ("❌" if required else "⚠️")
    req_text = "REQUERIDO" if required else "OPCIONAL"
    print(f"{status} {dirpath}/ - {req_text}")
    return exists

def get_file_size(filepath):
    """Obtiene el tamaño de un archivo en MB"""
    if Path(filepath).exists():
        size_mb = Path(filepath).stat().st_size / (1024 * 1024)
        return f"{size_mb:.2f} MB"
    return "N/A"

def main():
    print("="*60)
    print("🔍 VERIFICACIÓN PRE-DEPLOYMENT PARA STREAMLIT CLOUD")
    print("="*60)
    
    all_ok = True
    
    # Archivos principales
    print("\n📄 Archivos principales:")
    all_ok &= check_file("app.py", required=True)
    all_ok &= check_file("requirements.txt", required=True)
    all_ok &= check_file("packages.txt", required=True)
    all_ok &= check_file("README.md", required=True)
    all_ok &= check_file("config.json", required=False)
    
    # Configuración de Streamlit
    print("\n⚙️ Configuración de Streamlit:")
    all_ok &= check_directory(".streamlit", required=True)
    all_ok &= check_file(".streamlit/config.toml", required=True)
    
    # Código fuente
    print("\n💻 Código fuente:")
    all_ok &= check_directory("src", required=True)
    all_ok &= check_file("src/detector.py", required=True)
    all_ok &= check_file("src/preprocessing.py", required=True)
    all_ok &= check_file("src/logic.py", required=True)
    all_ok &= check_file("src/utils.py", required=True)
    
    # Modelo
    print("\n🧠 Modelo de ML:")
    model_exists = check_file("models/modelo_seguridad_v4.keras", required=True)
    if model_exists:
        size = get_file_size("models/modelo_seguridad_v4.keras")
        print(f"   Tamaño del modelo: {size}")
        
        # Verificar que no sea muy grande
        size_float = float(size.split()[0])
        if size_float > 100:
            print("   ⚠️ ADVERTENCIA: Modelo muy grande (>100MB)")
            print("   Streamlit Cloud Free tiene límite de 1GB total")
    
    all_ok &= model_exists
    
    # Datos (opcional para demo)
    print("\n📊 Datos de prueba (opcional):")
    check_directory("data", required=False)
    check_directory("data/raw", required=False)
    
    # Tests
    print("\n🧪 Tests:")
    check_directory("tests", required=False)
    check_file("tests/test_modelo.py", required=False)
    check_file("tests/test_logic.py", required=False)
    
    # Verificar contenido de requirements.txt
    print("\n📦 Verificando requirements.txt:")
    if Path("requirements.txt").exists():
        with open("requirements.txt", "r") as f:
            content = f.read()
            
        # Verificaciones importantes
        checks = {
            "tensorflow-cpu": "tensorflow-cpu" in content,
            "opencv-python-headless": "opencv-python-headless" in content or "opencv-python" in content,
            "streamlit": "streamlit" in content,
            "numpy": "numpy" in content,
            "pillow": "pillow" in content.lower(),
        }
        
        for package, found in checks.items():
            status = "✅" if found else "❌"
            print(f"   {status} {package}")
            all_ok &= found
    
    # Verificar packages.txt
    print("\n📦 Verificando packages.txt:")
    if Path("packages.txt").exists():
        with open("packages.txt", "r") as f:
            content = f.read()
            lines = [l.strip() for l in content.split("\n") if l.strip()]
            
        print(f"   Dependencias del sistema: {len(lines)}")
        for package in lines:
            print(f"   - {package}")
    else:
        print("   ❌ Archivo no encontrado")
        all_ok = False
    
    # Resumen final
    print("\n" + "="*60)
    if all_ok:
        print("✅ TODO LISTO PARA DEPLOYMENT")
        print("\n📋 Próximos pasos:")
        print("1. git add -A")
        print("2. git commit -m 'Preparado para Streamlit Cloud'")
        print("3. git push origin main")
        print("4. Ir a https://share.streamlit.io/")
        print("5. Deploy!")
        print("\n🔗 Ver DEPLOYMENT.md para instrucciones detalladas")
    else:
        print("❌ HAY PROBLEMAS QUE RESOLVER")
        print("\nRevisa los errores marcados con ❌ arriba")
        sys.exit(1)
    
    print("="*60)

if __name__ == "__main__":
    main()
