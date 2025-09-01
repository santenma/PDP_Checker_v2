# streamlit_app.py - VERSIÓN REFACTORIZADA Y OPTIMIZADA

import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# Importar TODA la funcionalidad desde core
from core import (
    # HTTP
    fetch_html,
    fetch_many_html,
    # PDP parsing
    assemble_product,
    parse_structured_product,
    clean_text,
    # Shopping
    search_shopping,
    analyze_shopping_data,
    # Matching
    MatchResult,
    match_greedy,
    # Normalización
    parse_price,
    normalize_specs,
    # Insights
    build_insights_report,
    price_kpis_from_matches,
)

# Opcionales para visualización
try:
    from wordcloud import WordCloud
    WORDCLOUD_OK = True
except ImportError:
    WORDCLOUD_OK = False

# ------------------------------------------------------
# Configuración y logging
# ------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Análisis Competitivo de Productos",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------
# Gestión de estado y configuración
# ------------------------------------------------------
class AppConfig:
    """Configuración centralizada de la aplicación"""
    
    def __init__(self):
        self.zenrows_key = self._get_zenrows_key()
        self.country = "es"
        self.force_zenrows = False
        self.js_render = True
        self.timeout = 30
        self.retries = 2
        self.min_interval = 1.0
        self.match_threshold = 70
        
    @staticmethod
    def _get_zenrows_key() -> Optional[str]:
        """Obtiene API key con prioridad: secrets > env"""
        try:
            return st.secrets.get("ZENROWS_API_KEY")
        except Exception:
            return os.getenv("ZENROWS_API_KEY")
    
    @property
    def has_zenrows(self) -> bool:
        return bool(self.zenrows_key)

@st.cache_resource
def get_config() -> AppConfig:
    """Singleton de configuración"""
    return AppConfig()

# ------------------------------------------------------
# Servicios de negocio (usando core)
# ------------------------------------------------------
class ProductAnalysisService:
    """Servicio principal para análisis de productos"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        
    def extract_pdps(self, urls: List[str], progress_callback=None) -> List[Dict[str, Any]]:
        """
        Extrae información de múltiples PDPs en paralelo
        """
        if not urls:
            return []
        
        # Usar fetch_many_html para paralelización
        logger.info(f"Extrayendo {len(urls)} PDPs...")
        
        fetch_kwargs = {
            "force_zenrows": self.config.force_zenrows,
            "js_render": self.config.js_render,
            "timeout": self.config.timeout,
            "retries": self.config.retries,
            "min_interval_sec": self.config.min_interval,
        }
        
        # Extraer HTML en paralelo
        url_html_pairs = fetch_many_html(urls, concurrency=4, per_request_kwargs=fetch_kwargs)
        
        # Procesar cada HTML
        products = []
        for i, (url, html) in enumerate(url_html_pairs):
            if progress_callback:
                progress_callback(i + 1, len(urls), url)
            
            if html:
                try:
                    product = assemble_product(url, html)
                    if product and product.get("title"):
                        # Añadir domain para mejor identificación
                        product["domain"] = urlparse(url).netloc
                        products.append(product)
                    else:
                        logger.warning(f"No se pudo extraer producto de: {url}")
                except Exception as e:
                    logger.error(f"Error procesando {url}: {e}")
            else:
                logger.warning(f"Sin HTML para: {url}")
                
        return products
    
    def search_market(self, query: str, num_results: int = 20) -> Tuple[List[Dict], Optional[str]]:
        """
        Busca productos en el mercado (Google Shopping + Web)
        """
        logger.info(f"Buscando en mercado: {query}")
        return search_shopping(
            query=query,
            country=self.config.country,
            num_results=num_results,
            force_zenrows=self.config.force_zenrows
        )
    
    def match_products(self, pdps: List[Dict], shopping: List[Dict]) -> List[MatchResult]:
        """
        Empareja PDPs con productos del mercado
        """
        logger.info(f"Emparejando {len(pdps)} PDPs con {len(shopping)} productos del mercado")
        return match_greedy(
            pdps=pdps,
            shops=shopping,
            threshold=self.config.match_threshold,
            one_to_one=True,
            prefer_price_proximity=True
        )
    
    def generate_insights(self, pdps: List[Dict], shopping: List[Dict], 
                         matches: List[MatchResult]) -> Dict[str, Any]:
        """
        Genera informe completo de insights
        """
        logger.info("Generando insights...")
        return build_insights_report(
            pdps=pdps,
            shopping_items=shopping,
            matches=matches,
            reference_idx=0,  # Primera URL como referencia
            target_currency="EUR"
        )

# ------------------------------------------------------
# Componentes UI
# ------------------------------------------------------
class UIComponents:
    """Componentes reutilizables de UI"""
    
    @staticmethod
    def render_header():
        """Header con estilos"""
        st.markdown("""
        <style>
            .main-header {
                font-size: 2.5rem;
                font-weight: 700;
                background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-align: center;
                margin-bottom: 2rem;
            }
            .metric-card {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 1rem;
                text-align: center;
            }
            .warning-box {
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 1rem;
                border-radius: 5px;
                margin: 1rem 0;
            }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<h1 class="main-header">🎯 Análisis Competitivo de Productos</h1>', 
                   unsafe_allow_html=True)
        st.caption("Análisis avanzado PDP vs Mercado con extracción robusta e insights automáticos")
    
    @staticmethod
    def render_config_sidebar(config: AppConfig):
        """Sidebar de configuración"""
        st.sidebar.header("⚙️ Configuración")
        
        # Estado de Zenrows
        if config.has_zenrows:
            st.sidebar.success("✅ Zenrows configurado")
        else:
            st.sidebar.warning("⚠️ Configura ZENROWS_API_KEY en Secrets")
        
        st.sidebar.markdown("---")
        
        # Configuración básica
        config.country = st.sidebar.selectbox(
            "🌍 País/Idioma", 
            ["es", "en", "fr", "de", "it"], 
            index=0
        )
        
        config.match_threshold = st.sidebar.slider(
            "🎯 Umbral de matching",
            50, 95, 70, 5,
            help="Score mínimo para considerar dos productos iguales"
        )
        
        # Configuración avanzada
        with st.sidebar.expander("⚡ Configuración avanzada"):
            config.force_zenrows = st.checkbox(
                "Forzar Zenrows",
                value=False,
                help="Usar siempre Zenrows en lugar de intentar directo"
            )
            
            config.js_render = st.checkbox(
                "JavaScript rendering",
                value=True,
                help="Renderizar JS (más lento pero más completo)"
            )
            
            config.timeout = st.slider(
                "Timeout (seg)",
                10, 60, 30, 5
            )
            
            config.retries = st.slider(
                "Reintentos",
                0, 5, 2, 1
            )
            
            config.min_interval = st.slider(
                "Intervalo entre requests (seg)",
                0.5, 5.0, 1.0, 0.5
            )
        
        st.sidebar.markdown("---")
        st.sidebar.caption("v4.0 • Core Integration • Parallel Processing")
    
    @staticmethod
    def render_pdp_metrics(pdps: List[Dict]):
        """Métricas de PDPs extraídos"""
        if not pdps:
            return
            
        df = pd.DataFrame(pdps)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📄 PDPs extraídos", len(pdps))
        
        with col2:
            with_price = df['price'].notna().sum()
            st.metric("💰 Con precio", f"{with_price}/{len(pdps)}")
        
        with col3:
            avg_specs = df['specifications'].apply(lambda x: len(x) if isinstance(x, dict) else 0).mean()
            st.metric("📊 Specs promedio", f"{avg_specs:.1f}")
        
        with col4:
            brands = df['brand'].nunique()
            st.metric("🏷️ Marcas únicas", brands)
    
    @staticmethod
    def render_shopping_metrics(shopping: List[Dict], analysis: Dict):
        """Métricas de productos del mercado"""
        if not shopping:
            return
            
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🛒 Productos encontrados", len(shopping))
        
        if analysis.get("price_stats"):
            stats = analysis["price_stats"]
            with col2:
                st.metric("💵 Precio mín", f"{stats['min']:.2f}€")
            with col3:
                st.metric("💸 Precio máx", f"{stats['max']:.2f}€")
            with col4:
                st.metric("📊 Precio medio", f"{stats['avg']:.2f}€")

# ------------------------------------------------------
# Tabs de la aplicación
# ------------------------------------------------------
class PDPAnalysisTab:
    """Tab de análisis de PDPs"""
    
    def __init__(self, service: ProductAnalysisService):
        self.service = service
    
    def render(self):
        st.subheader("🔗 Extracción de PDPs")
        
        # Input de URLs
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**🎯 Producto de referencia**")
            ref_url = st.text_input(
                "Tu producto (opcional)",
                placeholder="https://tu-tienda.com/producto",
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown("**🔍 Productos competidores**")
            comp_urls = st.text_area(
                "URLs de competidores (una por línea)",
                placeholder="https://amazon.es/...\nhttps://pccomponentes.com/...\nhttps://mediamarkt.es/...",
                height=100,
                label_visibility="collapsed"
            )
        
        # Botón de análisis
        if st.button("🚀 Extraer información de productos", type="primary", use_container_width=True):
            # Preparar URLs
            urls = []
            if ref_url and ref_url.strip():
                urls.append(ref_url.strip())
            urls.extend([u.strip() for u in comp_urls.splitlines() if u.strip()])
            urls = list(dict.fromkeys(urls))  # Eliminar duplicados
            
            if not urls:
                st.error("❌ Añade al menos una URL para analizar")
                return
            
            # Extraer PDPs con progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total, url):
                progress = current / total
                progress_bar.progress(progress)
                status_text.text(f"Extrayendo {current}/{total}: {urlparse(url).netloc}")
            
            with st.spinner("Extrayendo información de productos..."):
                pdps = self.service.extract_pdps(urls, progress_callback=update_progress)
            
            progress_bar.empty()
            status_text.empty()
            
            if pdps:
                st.success(f"✅ {len(pdps)} productos extraídos correctamente")
                st.session_state["pdp_data"] = pdps
                
                # Mostrar resultados
                UIComponents.render_pdp_metrics(pdps)
                
                # Tabla de resultados
                df = pd.DataFrame(pdps)[['title', 'price', 'brand', 'domain', 'url']]
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Detalles expandibles
                with st.expander("📋 Ver especificaciones extraídas"):
                    for i, pdp in enumerate(pdps):
                        st.markdown(f"**{i+1}. {pdp.get('title', 'Sin título')}**")
                        if pdp.get('specifications'):
                            specs_df = pd.DataFrame(
                                list(pdp['specifications'].items()),
                                columns=['Característica', 'Valor']
                            )
                            st.dataframe(specs_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("Sin especificaciones detectadas")
            else:
                st.error("❌ No se pudo extraer información de ninguna URL")

class MarketAnalysisTab:
    """Tab de análisis de mercado"""
    
    def __init__(self, service: ProductAnalysisService):
        self.service = service
    
    def render(self):
        st.subheader("🛒 Análisis de Mercado (Google Shopping)")
        
        # Búsqueda
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "¿Qué producto quieres buscar?",
                placeholder="Ej: iPhone 15 Pro Max 256GB",
                label_visibility="visible"
            )
        
        with col2:
            num_results = st.number_input(
                "Resultados",
                min_value=10,
                max_value=100,
                value=30,
                step=10
            )
        
        if st.button("🔎 Buscar en el mercado", type="primary", use_container_width=True):
            if not query:
                st.error("❌ Introduce un término de búsqueda")
                return
            
            with st.spinner(f"Buscando '{query}' en el mercado..."):
                products, error = self.service.search_market(query, num_results)
            
            if error:
                st.warning(f"⚠️ {error}")
            
            if products:
                st.success(f"✅ {len(products)} productos encontrados")
                st.session_state["shopping_products"] = products
                
                # Análisis del mercado
                analysis = analyze_shopping_data(products)
                st.session_state["market_analysis"] = analysis
                
                # Métricas
                UIComponents.render_shopping_metrics(products, analysis)
                
                # Visualizaciones
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribución por tienda
                    if analysis.get("sources"):
                        sources_df = pd.DataFrame(
                            list(analysis["sources"].items()),
                            columns=["Tienda", "Productos"]
                        ).sort_values("Productos", ascending=False).head(10)
                        
                        fig = px.bar(
                            sources_df,
                            x="Productos",
                            y="Tienda",
                            orientation="h",
                            title="Top 10 Tiendas",
                            color="Productos",
                            color_continuous_scale="Viridis"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Distribución de precios
                    if analysis.get("price_stats"):
                        prices = []
                        for p in products:
                            amt, _ = parse_price(p.get("price", ""))
                            if amt and 0.01 < amt < 100000:
                                prices.append(amt)
                        
                        if prices:
                            fig = px.histogram(
                                x=prices,
                                nbins=20,
                                title="Distribución de Precios",
                                labels={"x": "Precio (€)", "y": "Frecuencia"}
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                
                # Tabla de productos
                with st.expander("📊 Ver todos los productos encontrados"):
                    products_df = pd.DataFrame(products)
                    st.dataframe(products_df, use_container_width=True, hide_index=True)

class InsightsTab:
    """Tab de insights y matching"""
    
    def __init__(self, service: ProductAnalysisService):
        self.service = service
    
    def render(self):
        st.subheader("🔗 Matching & Insights Competitivos")
        
        pdps = st.session_state.get("pdp_data", [])
        shopping = st.session_state.get("shopping_products", [])
        
        if not pdps:
            st.info("💡 Primero extrae PDPs en la pestaña 'Extracción de PDPs'")
            return
        
        if not shopping:
            st.info("💡 Primero busca productos en la pestaña 'Análisis de Mercado'")
            return
        
        # Realizar matching
        if st.button("🎯 Ejecutar análisis competitivo", type="primary", use_container_width=True):
            with st.spinner("Emparejando productos y generando insights..."):
                # Matching
                matches = self.service.match_products(pdps, shopping)
                st.session_state["matches"] = matches
                
                # Insights
                insights = self.service.generate_insights(pdps, shopping, matches)
                st.session_state["insights"] = insights
            
            st.success("✅ Análisis completado")
        
        # Mostrar resultados si existen
        if "insights" in st.session_state:
            insights = st.session_state["insights"]
            
            # KPIs principales
            st.markdown("### 📊 KPIs Principales")
            
            price_data = insights.get("price", {})
            if price_data and "price_index_stats" in price_data and price_data["price_index_stats"]:
                stats = price_data["price_index_stats"]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "📈 Price Index Promedio",
                        f"{stats['avg']:.2f}",
                        help="1.0 = mismo precio, >1.0 = más caro, <1.0 = más barato"
                    )
                with col2:
                    st.metric("🎯 Match Rate", f"{price_data.get('match_rate', 0):.1%}")
                with col3:
                    outliers = price_data.get("outliers", {})
                    st.metric("⚠️ Outliers", f"{outliers.get('above', 0) + outliers.get('below', 0)}")
                with col4:
                    st.metric("💰 Rango PI", f"{stats['min']:.2f} - {stats['max']:.2f}")
            
            # Tabla de matching detallada
            st.markdown("### 🔗 Detalle de Matching")
            
            if "rows" in price_data:
                match_df = price_data["rows"]
                if isinstance(match_df, pd.DataFrame):
                    # Formatear columnas
                    display_df = match_df[[
                        'pdp_title', 'pdp_price_amount', 
                        'shop_title', 'shop_price_amount',
                        'match_score', 'price_index'
                    ]].copy()
                    
                    display_df.columns = [
                        'Producto PDP', 'Precio PDP',
                        'Producto Mercado', 'Precio Mercado',
                        'Score Match', 'Price Index'
                    ]
                    
                    # Aplicar estilos condicionales
                    def highlight_price_index(val):
                        if pd.isna(val):
                            return ''
                        if val > 1.1:
                            return 'background-color: #ffcccc'  # Rojo claro
                        elif val < 0.9:
                            return 'background-color: #ccffcc'  # Verde claro
                        return ''
                    
                    styled_df = display_df.style.applymap(
                        highlight_price_index,
                        subset=['Price Index']
                    ).format({
                        'Precio PDP': '€{:.2f}',
                        'Precio Mercado': '€{:.2f}',
                        'Score Match': '{:.0f}',
                        'Price Index': '{:.2f}'
                    }, na_rep='-')
                    
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Visualizaciones
            st.markdown("### 📈 Visualizaciones")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Scatter plot Price Index vs Match Score
                if "rows" in price_data and isinstance(price_data["rows"], pd.DataFrame):
                    df = price_data["rows"].dropna(subset=['price_index', 'match_score'])
                    if not df.empty:
                        fig = px.scatter(
                            df,
                            x='match_score',
                            y='price_index',
                            hover_data=['pdp_title', 'shop_title'],
                            title='Price Index vs Match Score',
                            labels={'match_score': 'Match Score', 'price_index': 'Price Index'},
                            color='price_index',
                            color_continuous_scale='RdYlGn_r'
                        )
                        fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Cobertura de especificaciones
                specs_data = insights.get("specs", {})
                if specs_data and "coverage_weighted" in specs_data:
                    coverage = specs_data["coverage_weighted"] * 100
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=coverage,
                        title={'text': "Cobertura de Especificaciones"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "gray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Gaps detectados
            st.markdown("### 🎯 Oportunidades Detectadas")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Especificaciones faltantes
                if specs_data and "missing_keys" in specs_data:
                    missing = specs_data["missing_keys"][:5]
                    if missing:
                        st.markdown("**📋 Especificaciones faltantes (top 5):**")
                        for item in missing:
                            st.markdown(f"- {item['key']} (peso: {item['weight']:.1f})")
            
            with col2:
                # Variantes faltantes
                variants_data = insights.get("variants", {})
                if variants_data and "missing_variants" in variants_data:
                    missing_vars = variants_data["missing_variants"]
                    if missing_vars:
                        st.markdown("**🎨 Variantes no cubiertas:**")
                        for var in missing_vars[:5]:
                            st.markdown(f"- {var}")

class ExportTab:
    """Tab de exportación de datos"""
    
    def render(self):
        st.subheader("📤 Exportar Resultados")
        
        # Verificar datos disponibles
        has_pdp = "pdp_data" in st.session_state
        has_shopping = "shopping_products" in st.session_state
        has_insights = "insights" in st.session_state
        
        if not any([has_pdp, has_shopping, has_insights]):
            st.info("💡 No hay datos para exportar. Realiza primero un análisis.")
            return
        
        st.markdown("### 📊 Datos disponibles para exportar")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if has_pdp:
                st.markdown("**🔗 PDPs Extraídos**")
                pdp_df = pd.DataFrame(st.session_state["pdp_data"])
                st.markdown(f"- {len(pdp_df)} productos")
                st.markdown(f"- {len(pdp_df.columns)} campos")
                
                csv = pdp_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Descargar PDPs (CSV)",
                    csv,
                    f"pdps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
        
        with col2:
            if has_shopping:
                st.markdown("**🛒 Productos del Mercado**")
                shop_df = pd.DataFrame(st.session_state["shopping_products"])
                st.markdown(f"- {len(shop_df)} productos")
                st.markdown(f"- {len(shop_df.columns)} campos")
                
                csv = shop_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Descargar Mercado (CSV)",
                    csv,
                    f"mercado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
        
        with col3:
            if has_insights:
                st.markdown("**📈 Insights Completos**")
                insights = st.session_state["insights"]
                
                # Convertir insights a formato exportable
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'config': {
                        'match_threshold': get_config().match_threshold,
                        'country': get_config().country
                    }
                }
                
                # Extraer datos principales
                if "price" in insights and "rows" in insights["price"]:
                    if isinstance(insights["price"]["rows"], pd.DataFrame):
                        export_data['matching'] = insights["price"]["rows"].to_dict('records')
                
                # KPIs
                if "price" in insights:
                    export_data['kpis'] = {
                        'match_rate': insights["price"].get("match_rate"),
                        'price_index_stats': insights["price"].get("price_index_stats"),
                        'outliers': insights["price"].get("outliers")
                    }
                
                # Exportar como JSON
                import json
                json_str = json.dumps(export_data, indent=2, default=str)
                st.download_button(
                    "⬇️ Descargar Insights (JSON)",
                    json_str,
                    f"insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json",
                    use_container_width=True
                )
        
        # Exportación combinada
        if has_pdp and has_shopping and has_insights:
            st.markdown("### 📦 Exportación Completa")
            
            if st.button("📄 Generar Reporte Completo (Excel)", use_container_width=True):
                try:
                    # Crear Excel con múltiples hojas
                    from io import BytesIO
                    output = BytesIO()
                    
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        # PDPs
                        if has_pdp:
                            pd.DataFrame(st.session_state["pdp_data"]).to_excel(
                                writer, sheet_name='PDPs', index=False
                            )
# Mercado
                        if has_shopping:
                            pd.DataFrame(st.session_state["shopping_products"]).to_excel(
                                writer, sheet_name='Mercado', index=False
                            )
                        
                        # Matching
                        if "insights" in st.session_state:
                            insights = st.session_state["insights"]
                            if "price" in insights and "rows" in insights["price"]:
                                if isinstance(insights["price"]["rows"], pd.DataFrame):
                                    insights["price"]["rows"].to_excel(
                                        writer, sheet_name='Matching', index=False
                                    )
                        
                        # KPIs
                        kpis_data = []
                        if "insights" in st.session_state:
                            insights = st.session_state["insights"]
                            
                            # Price KPIs
                            if "price" in insights:
                                kpis_data.append({
                                    'Métrica': 'Match Rate',
                                    'Valor': f"{insights['price'].get('match_rate', 0):.1%}"
                                })
                                
                                if "price_index_stats" in insights["price"] and insights["price"]["price_index_stats"]:
                                    stats = insights["price"]["price_index_stats"]
                                    kpis_data.extend([
                                        {'Métrica': 'Price Index Promedio', 'Valor': f"{stats['avg']:.2f}"},
                                        {'Métrica': 'Price Index Mínimo', 'Valor': f"{stats['min']:.2f}"},
                                        {'Métrica': 'Price Index Máximo', 'Valor': f"{stats['max']:.2f}"},
                                        {'Métrica': 'Price Index Mediana', 'Valor': f"{stats['median']:.2f}"},
                                    ])
                            
                            # Specs KPIs
                            if "specs" in insights and "coverage_weighted" in insights["specs"]:
                                kpis_data.append({
                                    'Métrica': 'Cobertura de Especificaciones',
                                    'Valor': f"{insights['specs']['coverage_weighted'] * 100:.1f}%"
                                })
                        
                        if kpis_data:
                            pd.DataFrame(kpis_data).to_excel(
                                writer, sheet_name='KPIs', index=False
                            )
                    
                    # Preparar descarga
                    output.seek(0)
                    st.download_button(
                        label="⬇️ Descargar Reporte Excel",
                        data=output,
                        file_name=f"reporte_completo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                    st.success("✅ Reporte generado correctamente")
                    
                except Exception as e:
                    st.error(f"❌ Error generando el reporte: {str(e)}")
                    logger.error(f"Error en exportación Excel: {e}")

# ------------------------------------------------------
# Aplicación principal
# ------------------------------------------------------
def main():
    """Función principal de la aplicación"""
    
    # Configuración
    config = get_config()
    
    # Header
    UIComponents.render_header()
    
    # Sidebar
    UIComponents.render_config_sidebar(config)
    
    # Servicio principal
    service = ProductAnalysisService(config)
    
    # Información inicial
    if not config.has_zenrows:
        st.warning("""
        ⚠️ **Zenrows API Key no configurada**
        
        Para obtener mejores resultados y evitar bloqueos:
        1. Obtén una API key gratuita en [zenrows.com](https://zenrows.com)
        2. Añádela en Streamlit Cloud: Settings → Secrets → `ZENROWS_API_KEY = "tu_key"`
        
        La aplicación funcionará con capacidades limitadas sin la API key.
        """)
    
    # Tabs principales
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔗 Extracción de PDPs",
        "🛒 Análisis de Mercado", 
        "🎯 Matching & Insights",
        "📤 Exportar Datos",
        "📚 Ayuda"
    ])
    
    with tab1:
        pdp_tab = PDPAnalysisTab(service)
        pdp_tab.render()
    
    with tab2:
        market_tab = MarketAnalysisTab(service)
        market_tab.render()
    
    with tab3:
        insights_tab = InsightsTab(service)
        insights_tab.render()
    
    with tab4:
        export_tab = ExportTab()
        export_tab.render()
    
    with tab5:
        render_help_tab()
    
    # Footer
    st.markdown("---")
    st.caption("""
    **PDP Analyzer v4.0** | Powered by Core Integration | 
    [GitHub](https://github.com/tu-usuario/tu-repo) | 
    [Reportar problemas](https://github.com/tu-usuario/tu-repo/issues)
    """)

def render_help_tab():
    """Tab de ayuda y documentación"""
    st.subheader("📚 Guía de Uso")
    
    with st.expander("🎯 ¿Qué es esta herramienta?"):
        st.markdown("""
        **PDP Analyzer** es una herramienta profesional de análisis competitivo que permite:
        
        - **Extraer información** de páginas de producto (PDPs) de cualquier e-commerce
        - **Analizar el mercado** mediante Google Shopping
        - **Comparar productos** automáticamente usando algoritmos de matching
        - **Generar insights** sobre posicionamiento de precio y gaps de contenido
        - **Exportar resultados** en múltiples formatos
        
        La herramienta utiliza técnicas avanzadas de web scraping y análisis de datos estructurados.
        """)
    
    with st.expander("🚀 Inicio Rápido"):
        st.markdown("""
        ### Flujo básico de trabajo:
        
        1. **Extracción de PDPs** 
           - Añade la URL de tu producto (opcional)
           - Añade URLs de competidores
           - Haz clic en "Extraer información"
        
        2. **Análisis de Mercado**
           - Busca tu producto en Google Shopping
           - Ajusta el número de resultados según necesites
           - Analiza la distribución de precios y tiendas
        
        3. **Matching & Insights**
           - La herramienta empareja automáticamente PDPs con productos del mercado
           - Calcula el Price Index (tu precio vs mercado)
           - Identifica gaps en especificaciones y variantes
        
        4. **Exportar**
           - Descarga los datos en CSV, JSON o Excel
           - Genera reportes completos con todas las métricas
        """)
    
    with st.expander("⚙️ Configuración Avanzada"):
        st.markdown("""
        ### Parámetros configurables:
        
        - **País/Idioma**: Afecta a las búsquedas en Google Shopping
        - **Umbral de Matching**: Score mínimo para considerar dos productos iguales (70-95)
        - **Forzar Zenrows**: Usar siempre proxy premium (más lento pero más fiable)
        - **JavaScript Rendering**: Necesario para sitios con contenido dinámico
        - **Timeout**: Tiempo máximo de espera por petición
        - **Reintentos**: Número de intentos si falla una petición
        - **Intervalo**: Tiempo entre peticiones para evitar rate limiting
        """)
    
    with st.expander("📊 Interpretación de Métricas"):
        st.markdown("""
        ### KPIs principales:
        
        **Price Index**
        - `1.0` = Tu precio es igual al mercado
        - `> 1.0` = Tu precio es superior al mercado
        - `< 1.0` = Tu precio es inferior al mercado
        - Ejemplo: `1.15` = Tu precio es 15% más caro
        
        **Match Score**
        - `0-100` = Similitud entre productos
        - `> 90` = Match muy probable
        - `70-90` = Match probable
        - `< 70` = Match poco probable
        
        **Coverage de Especificaciones**
        - % de características clave que tu producto tiene vs competencia
        - Identifica qué información falta en tu PDP
        
        **Outliers**
        - Productos con precios anormalmente altos o bajos
        - Útil para identificar errores o estrategias agresivas
        """)
    
    with st.expander("🐛 Solución de Problemas"):
        st.markdown("""
        ### Problemas comunes:
        
        **"No se pudo extraer contenido"**
        - El sitio puede estar bloqueando requests
        - Solución: Configura Zenrows API key
        
        **"Sin resultados en Google Shopping"**
        - La búsqueda es demasiado específica
        - Solución: Usa términos más genéricos
        
        **"Match Score bajo"**
        - Los títulos de productos son muy diferentes
        - Solución: Ajusta el umbral de matching
        
        **"Error al generar reporte"**
        - Puede faltar la librería xlsxwriter
        - Solución: Asegúrate de tener pandas con soporte Excel
        """)
    
    with st.expander("🔒 Privacidad y Límites"):
        st.markdown("""
        ### Información importante:
        
        - **Privacidad**: No almacenamos datos entre sesiones
        - **Rate Limiting**: Respetamos los límites de los sitios web
        - **Uso Responsable**: Esta herramienta es para análisis competitivo legítimo
        - **Zenrows**: Los límites dependen de tu plan (free = 1000 requests/mes)
        
        ### Limitaciones técnicas:
        
        - Máximo 100 URLs por análisis
        - Timeout de 60 segundos por petición
        - Algunos sitios pueden requerir configuración especial
        """)
    
    # Mostrar estado del sistema
    st.markdown("### 🔧 Estado del Sistema")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📦 Versión", "4.0.0")
        
    with col2:
        config = get_config()
        status = "✅ Configurado" if config.has_zenrows else "⚠️ No configurado"
        st.metric("🔑 Zenrows", status)
    
    with col3:
        # Verificar módulos opcionales
        modules = []
        try:
            import extruct
            modules.append("✅ extruct")
        except:
            modules.append("❌ extruct")
        
        try:
            from rapidfuzz import fuzz
            modules.append("✅ rapidfuzz")
        except:
            modules.append("❌ rapidfuzz")
        
        try:
            from price_parser import Price
            modules.append("✅ price-parser")
        except:
            modules.append("❌ price-parser")
        
        st.metric("📚 Módulos", " | ".join(modules))

# ------------------------------------------------------
# Punto de entrada
# ------------------------------------------------------
if __name__ == "__main__":
    # Configurar logging
    if os.getenv("DEBUG"):
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ejecutar aplicación
    try:
        main()
    except Exception as e:
        logger.error(f"Error crítico en la aplicación: {e}", exc_info=True)
        st.error(f"""
        ❌ **Error crítico en la aplicación**
        
        {str(e)}
        
        Por favor, reporta este error en [GitHub Issues](https://github.com/tu-usuario/tu-repo/issues)
        """)
