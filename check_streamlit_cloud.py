"""
Script para verificar el entorno de Streamlit Cloud
"""

import sys
import platform

def check_environment():
    print("=" * 50)
    print("VERIFICACIÓN DE ENTORNO")
    print("=" * 50)
    
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    print("\n" + "=" * 50)
    print("MÓDULOS CRÍTICOS")
    print("=" * 50)
    
    critical_modules = [
        'streamlit',
        'pandas',
        'numpy',
        'requests',
        'bs4',
        'plotly',
        'extruct',
        'rapidfuzz',
        'price_parser'
    ]
    
    for module in critical_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
    
    print("\nEntorno listo para ejecutar" if all(
        can_import(m) for m in critical_modules
    ) else "\n⚠️ Faltan módulos críticos")

def can_import(module_name):
    try:
        __import__(module_name)
        return True
    except:
        return False

if __name__ == "__main__":
    check_environment()
