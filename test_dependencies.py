# test_dependencies.py
"""Verificar que todas las dependencias críticas están instaladas y funcionan"""

import sys
import importlib
from typing import List, Tuple

def check_imports() -> List[Tuple[str, bool, str]]:
    """Verifica la disponibilidad de módulos críticos"""
    
    modules_to_check = [
        # Core
        ('streamlit', True, 'Framework principal'),
        ('pandas', True, 'Procesamiento de datos'),
        ('requests', True, 'HTTP client'),
        ('bs4', True, 'HTML parsing'),
        
        # Visualización
        ('plotly', True, 'Gráficos interactivos'),
        ('wordcloud', False, 'Nubes de palabras'),
        
        # Matching
        ('rapidfuzz', True, 'Fuzzy matching'),
        ('Levenshtein', False, 'Matching optimizado'),
        
        # Parsing
        ('price_parser', True, 'Parsing de precios'),
        ('babel', True, 'Internacionalización'),
        
        # Structured Data (CRÍTICO)
        ('extruct', True, 'Extracción de datos estructurados'),
        ('w3lib', True, 'Utilidades web'),
        ('rdflib', True, 'Procesamiento RDF'),
        ('html5lib', True, 'Parser HTML fallback'),
        
        # Export
        ('openpyxl', True, 'Excel export'),
        ('xlsxwriter', True, 'Excel formatting'),
        
        # Performance
        ('diskcache', False, 'Cache persistente'),
        ('aiohttp', False, 'Async HTTP'),
        ('tenacity', False, 'Retry avanzado'),
        
        # Optional
        ('lxml', False, 'Parser HTML rápido'),
    ]
    
    results = []
    for module_name, is_critical, description in modules_to_check:
        try:
            importlib.import_module(module_name)
            results.append((module_name, True, description))
        except ImportError:
            results.append((module_name, False, description))
            if is_critical:
                print(f"❌ CRÍTICO: {module_name} no está instalado - {description}")
    
    return results

def test_extruct_functionality():
    """Verifica que extruct funciona correctamente"""
    try:
        import extruct
        from w3lib.html import get_base_url
        
        # HTML de prueba con JSON-LD
        test_html = """
        <html>
        <head>
            <script type="application/ld+json">
            {
                "@context": "https://schema.org",
                "@type": "Product",
                "name": "Test Product",
                "price": "99.99"
            }
            </script>
        </head>
        </html>
        """
        
        data = extruct.extract(test_html, base_url="https://example.com")
        assert 'json-ld' in data
        assert len(data['json-ld']) > 0
        print("✅ extruct funciona correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en extruct: {e}")
        return False

def test_price_parser():
    """Verifica price-parser"""
    try:
        from price_parser import Price
        
        p = Price.fromstring("$99.99")
        assert p.amount_float == 99.99
        assert p.currency == "$"
        print("✅ price-parser funciona correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en price-parser: {e}")
        return False

def test_rapidfuzz():
    """Verifica rapidfuzz"""
    try:
        from rapidfuzz import fuzz
        
        score = fuzz.ratio("hello world", "hello word")
        assert 80 < score < 100
        print("✅ rapidfuzz funciona correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en rapidfuzz: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Verificando dependencias...")
    print("=" * 50)
    
    results = check_imports()
    
    # Resumen
    total = len(results)
    installed = sum(1 for _, status, _ in results if status)
    critical_missing = [name for name, status, _ in results if not status and name in [
        'streamlit', 'pandas', 'requests', 'bs4', 'extruct', 'w3lib', 
        'price_parser', 'rapidfuzz', 'plotly'
    ]]
    
    print("\n" + "=" * 50)
    print(f"Resumen: {installed}/{total} módulos instalados")
    
    if critical_missing:
        print(f"\n⚠️  Módulos críticos faltantes: {', '.join(critical_missing)}")
        print("Instala con: pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("\n✅ Todos los módulos críticos están instalados")
        
        # Tests funcionales
        print("\n" + "=" * 50)
        print("Ejecutando tests funcionales...")
        print("=" * 50)
        
        test_extruct_functionality()
        test_price_parser()
        test_rapidfuzz()
        
        print("\n✅ Sistema listo para funcionar")
