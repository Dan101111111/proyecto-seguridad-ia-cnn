"""
Script para organizar datasets de Roboflow en la estructura del proyecto
Copia todas las imágenes de train/ y valid/ a data/raw/[clase]/
"""
import os
import shutil
from pathlib import Path

print("="*70)
print("ORGANIZADOR DE DATASETS - Proyecto Seguridad IA")
print("="*70)

# Mapeo de nombres de dataset a nombres de carpeta
# Ajusta estos nombres según tus carpetas descargadas
DATASET_MAPPING = {
    # "nombre_carpeta_descargada": "nombre_carpeta_destino"
    "personas": "persona",
    "person": "persona",
    "people": "persona",
    
    "armas": "arma",
    "gun": "arma",
    "weapon": "arma",
    "guns": "arma",
    
    "mascaras": "mascara",
    "mask": "mascara",
    "masks": "mascara",
    "face-mask": "mascara",
    
    "gorros": "gorro",
    "gorro": "gorro",
    "hat": "gorro",
    "helmet": "gorro",
    "hats": "gorro",
    "helmets": "gorro",
    "casco": "gorro",
}

def encontrar_imagenes(directorio):
    """Encuentra todas las imágenes en un directorio"""
    extensiones = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    imagenes = []
    
    for root, dirs, files in os.walk(directorio):
        for file in files:
            if Path(file).suffix.lower() in extensiones:
                imagenes.append(os.path.join(root, file))
    
    return imagenes

def identificar_clase(nombre_carpeta):
    """Identifica a qué clase pertenece un dataset"""
    nombre_lower = nombre_carpeta.lower()
    
    # Buscar coincidencia exacta o parcial
    for key, clase in DATASET_MAPPING.items():
        if key in nombre_lower or nombre_lower in key:
            return clase
    
    return None

def organizar_dataset(ruta_dataset, clase_destino):
    """Organiza un dataset en la estructura del proyecto"""
    
    print(f"\n📁 Procesando: {os.path.basename(ruta_dataset)}")
    print(f"   → Destino: data/raw/{clase_destino}/")
    
    # Crear carpeta destino
    destino = f"data/raw/{clase_destino}"
    os.makedirs(destino, exist_ok=True)
    
    # Contar imágenes existentes para continuar numeración
    imagenes_existentes = [f for f in os.listdir(destino) if f.endswith(('.jpg', '.jpeg', '.png'))]
    contador = len(imagenes_existentes) + 1
    
    # Buscar imágenes en train/ y valid/
    carpetas_origen = ['train', 'valid', 'test']
    total_copiadas = 0
    
    for carpeta in carpetas_origen:
        ruta_carpeta = os.path.join(ruta_dataset, carpeta)
        
        if not os.path.exists(ruta_carpeta):
            continue
        
        print(f"   🔍 Buscando en {carpeta}/...")
        imagenes = encontrar_imagenes(ruta_carpeta)
        
        for img in imagenes:
            # Nuevo nombre: clase_001.jpg, clase_002.jpg, etc.
            extension = Path(img).suffix
            nuevo_nombre = f"{clase_destino}_{contador:03d}{extension}"
            ruta_destino = os.path.join(destino, nuevo_nombre)
            
            # Copiar imagen
            shutil.copy2(img, ruta_destino)
            contador += 1
            total_copiadas += 1
        
        if imagenes:
            print(f"      ✅ {len(imagenes)} imágenes de {carpeta}/")
    
    print(f"   📊 Total copiadas: {total_copiadas}")
    print(f"   📊 Total en carpeta: {contador - 1}")
    
    return total_copiadas

def main():
    print("\n🔍 Buscando datasets descargados...\n")
    
    # Solicitar ruta donde están los datasets
    print("Ingresa la ruta donde descargaste los 4 datasets")
    print("Ejemplo: C:\\Users\\Daniel\\Downloads\\datasets")
    print("O presiona Enter si están en la carpeta actual")
    
    ruta_base = input("\nRuta de datasets: ").strip() or "."
    
    if not os.path.exists(ruta_base):
        print(f"\n❌ ERROR: La ruta '{ruta_base}' no existe")
        return
    
    # Buscar carpetas de datasets
    carpetas = [d for d in os.listdir(ruta_base) 
                if os.path.isdir(os.path.join(ruta_base, d))]
    
    print(f"\n📂 Carpetas encontradas: {len(carpetas)}")
    
    # Organizar cada dataset
    total_imagenes = 0
    datasets_procesados = []
    
    for carpeta in carpetas:
        ruta_completa = os.path.join(ruta_base, carpeta)
        
        # Verificar si tiene estructura de Roboflow (train/valid/test)
        tiene_train = os.path.exists(os.path.join(ruta_completa, 'train'))
        tiene_valid = os.path.exists(os.path.join(ruta_completa, 'valid'))
        tiene_yaml = os.path.exists(os.path.join(ruta_completa, 'data.yaml'))
        
        if not (tiene_train or tiene_valid):
            print(f"\n⏭️  Saltando '{carpeta}' (no parece dataset de Roboflow)")
            continue
        
        # Identificar clase
        clase = identificar_clase(carpeta)
        
        if not clase:
            print(f"\n⚠️  No se pudo identificar la clase de '{carpeta}'")
            print("   Clases válidas: arma, gorro, mascara, persona")
            clase = input("   Ingresa la clase manualmente (o Enter para saltar): ").strip().lower()
            
            if not clase or clase not in ['arma', 'gorro', 'mascara', 'persona']:
                print("   ⏭️  Saltando...")
                continue
        
        # Organizar dataset
        num_imagenes = organizar_dataset(ruta_completa, clase)
        total_imagenes += num_imagenes
        datasets_procesados.append((carpeta, clase, num_imagenes))
    
    # Resumen final
    print("\n" + "="*70)
    print("📊 RESUMEN DE ORGANIZACIÓN")
    print("="*70)
    
    if datasets_procesados:
        for dataset, clase, num in datasets_procesados:
            print(f"   ✅ {dataset[:40]:<40} → {clase:<10} ({num} imágenes)")
    else:
        print("   ⚠️  No se procesaron datasets")
    
    print(f"\n   Total imágenes copiadas: {total_imagenes}")
    
    # Mostrar estado final de data/raw/
    print("\n" + "="*70)
    print("📁 ESTADO FINAL: data/raw/")
    print("="*70)
    
    for clase in ['arma', 'gorro', 'mascara', 'persona']:
        ruta = f"data/raw/{clase}"
        if os.path.exists(ruta):
            num_imgs = len([f for f in os.listdir(ruta) 
                           if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"   {clase:<10} : {num_imgs:>4} imágenes")
    
    print("\n" + "="*70)
    print("✅ ORGANIZACIÓN COMPLETADA")
    print("="*70)
    print("\n🚀 Próximo paso: Entrenar el modelo con más datos")
    print("   python entrenar_modelo_v4.py")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Proceso cancelado por el usuario")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
