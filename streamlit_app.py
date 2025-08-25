import os
import re
import json
import time
import random
import warnings
from datetime import datetime
from urllib.parse import urlparse, quote_plus, urlencode

import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from collections import Counter

# Visualización
import plotly.express as px
import plotly.graph_objects as go

# Opcionales (se activan si están en requirements; si no, hay fallback)
try:
    import extruct
    from w3lib.html import get_base_url
    EXSTRUCT_OK = True
except Exception:
    EXSTRUCT_OK = False

try:
    from price_parser import Price
    PRICE_PARSER_OK = True
except Exception:
    PRICE_PARSER_OK = False

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_OK = True
except Exception:
    RAPIDFUZZ_OK = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

# ------------------------------------------------------
# Configuración general
# ------------------------------------------------------
warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="Análisis Competitivo de Productos",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Detectar API Key de Zenrows (Streamlit secrets o env)
ZENROWS_KEY = st.secrets.get("ZENROWS_API_KEY", os.getenv("ZENROWS_API_KEY"))

# ------------------------------------------------------
# Utilidades comunes
# ------------------------------------------------------
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
]

def _ua():
    return {"User-Agent": random.choice(USER_AGENTS)}

def _looks_like_captcha(html: str) -> bool:
    if not html:
        return False
    text = html.lower()
    patterns = ["captcha", "unusual traffic", "verify you are a human", "bot detection"]
    return any(p in text for p in patterns)

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_html(url: str, force_zenrows: bool = False, js_render: bool = True, delay: float = 1.2) -> str:
    """
    Obtiene HTML con request directa; si hay 403/CAPTCHA o force_zenrows, usa Zenrows.
    """
    # 1) Intento directo
    if not force_zenrows:
        try:
            r = requests.get(url, headers=_ua(), timeout=25, allow_redirects=True)
            # Si 403 o pinta a CAPTCHA, saltar a Zenrows
            if r.status_code == 200 and not _looks_like_captcha(r.text):
                return r.text
        except Exception:
            pass

    # 2) Fallback Zenrows (si hay API Key)
    if not ZENROWS_KEY:
        # Sin key → devolvemos vacío para que el caller maneje el error
        return ""

    params = {
        "url": url,
        "premium_proxy": "true",
        "antibot": "true",
    }
    if js_render:
        params["js_render"] = "true"

    try:
        zr = requests.get(
            "https://api.zenrows.com/v1/?" + urlencode(params),
            headers=_ua(),
            auth=(ZENROWS_KEY, ""),
            timeout=40,
        )
        if zr.status_code == 200:
            time.sleep(delay)
            return zr.text
    except Exception:
        pass

    return ""

def parse_price_text(text: str):
    """Devuelve (amount_float, currency) si price-parser está disponible; si no, regex."""
    if not text:
        return None, None
    if PRICE_PARSER_OK:
        p = Price.fromstring(text)
        return p.amount_float, (p.currency or "€")
    # Regex fallback
    m = re.search(r"(\d+(?:[.,]\d{2})?)", text.replace(",", "."))
    return (float(m.group(1)) if m else None), None

def clean_text(s):
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()

# ------------------------------------------------------
# Parse PDP — structured + OG + heurísticos
# ------------------------------------------------------
def parse_structured_product(html: str, url: str) -> dict:
    if not EXSTRUCT_OK or not html:
        return {}
    try:
        base_url = get_base_url(html, url)
        data = extruct.extract(html, base_url=base_url, syntaxes=["json-ld", "microdata", "opengraph"])
        # JSON-LD
        for ld in data.get("json-ld", []):
            if isinstance(ld, dict):
                t = ld.get("@type")
                if t == "Product" or (isinstance(t, list) and "Product" in t):
                    return ld
        # Microdata
        for md in data.get("microdata", []):
            types = md.get("type", [])
            if any("Product" in t for t in types):
                props = md.get("properties", {})
                if props:
                    return props
        # OpenGraph
        og_list = data.get("opengraph", [])
        if og_list:
            og = {i.get("property", ""): i.get("content", "") for i in og_list if i.get("property")}
            return og or {}
    except Exception:
        return {}
    return {}

def parse_og(soup: BeautifulSoup) -> dict:
    out = {}
    for tag in soup.select("meta[property^='og:'], meta[name^='og:']"):
        prop = tag.get("property") or tag.get("name")
        content = tag.get("content")
        if prop and content:
            out[prop] = content
    return out

def extract_heuristics(soup: BeautifulSoup) -> dict:
    # Título
    title = ""
    if soup.select_one("h1"):
        title = clean_text(soup.select_one("h1").get_text())
    if not title and soup.title:
        title = clean_text(soup.title.get_text())

    # Descripción
    desc = ""
    mdesc = soup.select_one("meta[name='description'], meta[property='og:description']")
    if mdesc:
        desc = clean_text(mdesc.get("content", ""))
    if not desc:
        p = soup.select_one("p")
        if p:
            desc = clean_text(p.get_text())

    # Características
    features = []
    feature_selectors = [
        '[class*="feature"] li',
        '[class*="benefit"] li',
        '[class*="highlight"] li',
        '[class*="spec"] li',
        ".features li",
        ".benefits li",
        "div[class*='feature']",
    ]
    for sel in feature_selectors:
        for el in soup.select(sel):
            txt = clean_text(el.get_text())
            if txt and 10 <= len(txt) <= 500 and not re.match(r"^\d+$", txt):
                features.append(txt)
    # Unique
    seen = set()
    features = [f for f in features if not (f.lower() in seen or seen.add(f.lower()))][:50]

    # Especificaciones (tablas/dl)
    specs = {}
    for table in soup.select("table[class*='spec'], table[class*='tech'], table, dl[class*='spec']"):
        if table.name == "table":
            for tr in table.select("tr"):
                cells = tr.find_all(["td", "th"])
                if len(cells) >= 2:
                    k = clean_text(cells[0].get_text())
                    v = clean_text(cells[1].get_text())
                    if k and v and len(k) < 100 and len(v) < 200:
                        specs[k] = v
        if table.name == "dl":
            dts = table.find_all("dt")
            dds = table.find_all("dd")
            for dt, dd in zip(dts, dds):
                k = clean_text(dt.get_text())
                v = clean_text(dd.get_text())
                if k and v:
                    specs[k] = v

    # Categorías/breadcrumbs
    categories = []
    for sel in ["nav.breadcrumb a", ".breadcrumb a", "[itemprop='itemListElement'] a", "a[rel='breadcrumb']"]:
        for a in soup.select(sel):
            txt = clean_text(a.get_text())
            if txt and txt.lower() not in ("home", "inicio"):
                categories.append(txt)
    categories = list(dict.fromkeys(categories))[:10]

    # Filtros (heurístico)
    filters_ = []
    for sel in ["[class*='filter'] a", "[class*='facet'] a", "[class*='filter'] label", "[class*='facet'] label"]:
        for el in soup.select(sel):
            txt = clean_text(el.get_text())
            if txt and 2 < len(txt) < 40:
                filters_.append(txt)
    filters_ = list(dict.fromkeys(filters_))[:40]

    # Imágenes
    images = []
    for img in soup.select("img"):
        src = img.get("src") or img.get("data-src") or img.get("content")
        if src and src.startswith("http"):
            images.append(src)
    images = list(dict.fromkeys(images))[:10]

    # Precio (heurístico con parser opcional)
    price_text = ""
    for el in soup.select("[class*='price'], [class*='cost'], [class*='amount'], [id*='price'], span[itemprop='price'], meta[itemprop='price']"):
        txt = clean_text(el.get_text() or el.get("content") or "")
        if any(ch.isdigit() for ch in txt):
            price_text = txt
            break

    return {
        "title": title,
        "description": desc,
        "features": features,
        "specifications": specs,
        "categories": categories,
        "filters": filters_,
        "images": images,
        "raw_price": price_text,
    }

def assemble_product(url: str, html: str) -> dict:
    if not html:
        return {}
    soup = BeautifulSoup(html, "html.parser")
    structured = parse_structured_product(html, url) or {}
    og = parse_og(soup)

    # Merge con prioridad: structured > OG > heurísticos
    heur = extract_heuristics(soup)

    def pick(*opts):
        for o in opts:
            if isinstance(o, str) and o:
                return o
        return ""

    title = pick(
        structured.get("name"),
        structured.get("title"),
        og.get("og:title"),
        heur["title"],
    )
    description = pick(
        structured.get("description"),
        og.get("og:description"),
        heur["description"],
    )

    # Precio
    price_val, currency = None, None
    candidate_prices = []
    # Structured
    if isinstance(structured.get("offers"), dict):
        candidate_prices.append(str(structured["offers"].get("price", "")))
    if "raw_price" in heur and heur["raw_price"]:
        candidate_prices.append(heur["raw_price"])
    # OG meta price?
    candidate_prices.append(og.get("product:price:amount", ""))

    for prw in candidate_prices:
        price_val, currency = parse_price_text(prw)
        if price_val:
            break

    product_data = {
        "url": url,
        "domain": urlparse(url).netloc,
        "title": title,
        "description": description,
        "features": heur["features"],
        "specifications": heur["specifications"],
        "price": f"{price_val:.2f} {currency}" if price_val else (heur.get("raw_price") or ""),
        "filters": heur["filters"],
        "categories": heur["categories"],
        "images": heur["images"],
        "brand": structured.get("brand") if isinstance(structured.get("brand"), str) else (structured.get("brand", {}) or {}).get("name"),
        "sku": structured.get("sku"),
        "mpn": structured.get("mpn"),
        "gtin": structured.get("gtin13") or structured.get("gtin") or structured.get("gtin8") or structured.get("gtin12"),
        "rating": (structured.get("aggregateRating", {}) or {}).get("ratingValue") if isinstance(structured.get("aggregateRating"), dict) else None,
        "reviewCount": (structured.get("aggregateRating", {}) or {}).get("reviewCount") if isinstance(structured.get("aggregateRating"), dict) else None,
        "availability": (structured.get("offers", {}) or {}).get("availability") if isinstance(structured.get("offers"), dict) else None,
        "seller": (structured.get("offers", {}) or {}).get("seller") if isinstance(structured.get("offers"), dict) else None,
        "extracted_at": datetime.now().isoformat(),
    }
    return product_data

# ------------------------------------------------------
# Google Shopping (búsqueda free basada en HTML público)
# ------------------------------------------------------
class GoogleShoppingAnalyzer:
    def __init__(self, country="es"):
        self.country = country

    @st.cache_data(show_spinner=False, ttl=1800)
    def search_products_free(_self, query: str, num_results: int = 20):
        """
        Búsqueda simple: intenta Google Shopping; si falla o hay pocos resultados, usa búsqueda web.
        No usa APIs privadas; respeta límites; Zenrows si hay bloqueo.
        """
        if not query or not query.strip():
            return [], "Query vacío"

        products = []
        errors = []

        # 1) Intentar vertical Shopping
        q = quote_plus(query)
        shopping_url = f"https://www.google.com/search?tbm=shop&hl={_self.country}&q={q}"
        html = fetch_html(shopping_url, force_zenrows=False, js_render=True)
        if not html:
            # Forzar Zenrows si hay key
            html = fetch_html(shopping_url, force_zenrows=True, js_render=True)

        if html:
            products.extend(_self._parse_shopping_html(html))
        else:
            errors.append("Shopping: sin respuesta")

        # 2) Fallback búsqueda web si hay pocos
        if len(products) < 3:
            web_url = f"https://www.google.com/search?hl={_self.country}&q={q}"
            html2 = fetch_html(web_url, force_zenrows=False, js_render=False)
            if not html2:
                html2 = fetch_html(web_url, force_zenrows=True, js_render=False)
            if html2:
                products.extend(_self._parse_web_html(html2))
            else:
                errors.append("Web: sin respuesta")

        # Unificar y truncar
        seen = set()
        uniq = []
        for p in products:
            k = (p.get("title", ""), p.get("source", ""), p.get("link", ""))
            if k not in seen:
                seen.add(k)
                uniq.append(p)
        uniq = uniq[: num_results]

        return uniq, ("; ".join(errors) if errors else None)

    def _parse_shopping_html(self, html: str):
        soup = BeautifulSoup(html, "html.parser")
        out = []
        # Selectores conservadores para tarjetas de shopping
        cards = soup.select("div.sh-dgr__content, div.sh-pr__product-results, div.pslires")
        if not cards:
            cards = soup.select("div.sh-dlr__list-result, div.sh-dlr__content")
        for card in cards:
            title = clean_text(card.get_text(" ", strip=True)[:140])
            price_el = card.select_one("span.a8Pemb, span.OFFNJd, span.kHxwFf")
            price_txt = clean_text(price_el.get_text()) if price_el else ""
            amount, _cur = parse_price_text(price_txt)
            seller_el = card.select_one("div.aULzUe, div.aEZQsc")
            seller = clean_text(seller_el.get_text()) if seller_el else ""
            link_el = card.select_one("a[href]")
            link = "https://www.google.com" + link_el["href"] if link_el and link_el.get("href", "").startswith("/") else (link_el["href"] if link_el else "#")

            if title:
                out.append(
                    {
                        "title": title,
                        "price": f"{amount:.2f} €" if amount else price_txt or "N/A",
                        "source": seller or "Google Shopping",
                        "link": link,
                    }
                )
        return out

    def _parse_web_html(self, html: str):
        soup = BeautifulSoup(html, "html.parser")
        out = []
        for res in soup.select("div.g, div.MjjYud, div.tF2Cxc"):
            a = res.select_one("a[href]")
            title_el = res.select_one("h3")
            snip = res.select_one("div.VwiC3b")
            if a and title_el:
                out.append(
                    {
                        "title": clean_text(title_el.get_text()),
                        "price": "N/A",
                        "source": urlparse(a["href"]).netloc,
                        "link": a["href"],
                        "description": clean_text(snip.get_text()) if snip else "",
                    }
                )
        return out

    def analyze_shopping_data(self, products):
        if not products:
            return {"total_products": 0, "sources": {}, "price_ranges": None, "common_terms": Counter(), "has_data": False}

        analysis = {"total_products": len(products), "sources": {}, "price_ranges": None, "common_terms": Counter(), "has_data": True}

        # Fuentes
        for p in products:
            src = p.get("source", "Desconocido")
            analysis["sources"][src] = analysis["sources"].get(src, 0) + 1

        # Precios
        prices = []
        for p in products:
            amt, _cur = parse_price_text(p.get("price", ""))
            if amt and 0.01 < amt < 100000:
                prices.append(amt)
        if prices:
            prices_sorted = sorted(prices)
            median = prices_sorted[len(prices_sorted) // 2]
            analysis["price_ranges"] = {"min": min(prices), "max": max(prices), "avg": sum(prices) / len(prices), "median": median, "count": len(prices)}

        # Términos básicos
        all_text = " ".join([f"{p.get('title','')} {p.get('description','')}" for p in products])
        words = re.findall(r"\b[a-záéíóúñü0-9]{3,}\b", all_text.lower())
        stopwords = {"para", "con", "por", "del", "las", "los", "una", "uno", "desde", "hasta", "más", "muy", "todo", "todos", "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas"}
        filtered = [w for w in words if w not in stopwords]
        analysis["common_terms"] = Counter(filtered)
        return analysis

# ------------------------------------------------------
# Matching PDP ↔ Shopping
# ------------------------------------------------------
def simple_ratio(a: str, b: str) -> int:
    a = (a or "").lower()
    b = (b or "").lower()
    if RAPIDFUZZ_OK:
        return fuzz.token_set_ratio(a, b)
    # Fallback naive
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a or not set_b:
        return 0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return int(100 * inter / union)

def match_pdp_to_shopping(pdp_item: dict, shopping_items: list, threshold: int = 70):
    # 1) Clave dura
    for key in ["gtin", "mpn", "sku"]:
        if pdp_item.get(key):
            for s in shopping_items:
                if s.get(key) and s[key] == pdp_item[key]:
                    return s, 100
    # 2) Clave blanda (marca + modelo + título)
    target = " ".join(
        [
            str(pdp_item.get("brand") or ""),
            str(pdp_item.get("title") or ""),
        ]
    ).strip()
    best, best_score = None, -1
    for s in shopping_items:
        cand = " ".join([str(s.get("brand") or ""), str(s.get("title") or "")]).strip()
        score = simple_ratio(target, cand)
        if score > best_score:
            best, best_score = s, score
    if best_score >= threshold:
        return best, best_score
    return None, best_score

# ------------------------------------------------------
# UI
# ------------------------------------------------------
def header_css():
    st.markdown(
        """
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
      .gap-card {
          background: #f8f9fa;
          border-left: 4px solid #FF6B6B;
          padding: 1rem;
          margin: 0.5rem 0;
          border-radius: 0.25rem;
      }
    </style>
    """,
        unsafe_allow_html=True,
    )

def main():
    header_css()
    st.markdown('<h1 class="main-header">🎯 Análisis Competitivo de Productos</h1>', unsafe_allow_html=True)
    st.caption("PDP vs Mercado (Google Shopping) con extracción robusta y emparejamiento por similitud")

    # Ayuda
    with st.expander("📚 ¿Cómo usar esta herramienta?"):
        st.markdown(
            """
        1) **Pestaña PDP (URLs):** introduce la URL de tu producto y varias de competidores.  
        2) **Pestaña Shopping:** busca el producto en Google Shopping.  
        3) **Matching & Insights:** empareja cada PDP con el mejor resultado de Shopping y obtén KPIs (price index, gaps, etc.).  
        4) **Exportar:** descarga resultados.
        """
        )
        if not EXSTRUCT_OK:
            st.info("ℹ️ Tip: añade `extruct` y `w3lib` a requirements para extraer schema.org automáticamente.")
        if not PRICE_PARSER_OK:
            st.info("ℹ️ Tip: añade `price-parser` a requirements para parsear precios de forma robusta.")
        if not RAPIDFUZZ_OK:
            st.info("ℹ️ Tip: añade `rapidfuzz` a requirements para matching más preciso.")

    # Sidebar
    st.sidebar.header("⚙️ Configuración")
    top_n = st.sidebar.slider("📊 Top N resultados (Shopping)", 5, 50, 20, 5)
    delay = st.sidebar.slider("⏱️ Delay entre requests (seg)", 0.5, 5.0, 1.5, 0.5)
    rotate_headers = st.sidebar.checkbox("🔄 Rotar User-Agents", value=True)
    force_zen = st.sidebar.checkbox("🛡️ Forzar Zenrows para PDP", value=False)
    st.sidebar.markdown("---")
    country = st.sidebar.selectbox("🌍 País / idioma búsqueda", ["es", "en", "fr", "de", "it"], index=0)
    st.sidebar.markdown("---")
    st.sidebar.caption("Configura `ZENROWS_API_KEY` en **Secrets** para mayor robustez.")

    # Pestañas
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔗 PDP por URLs", "🛒 Google Shopping", "🔗 Matching & Insights", "📤 Exportar"]
    )

    # ---------------------------
    # TAB 1 — PDP por URLs
    # ---------------------------
    with tab1:
        st.subheader("🎯 Producto de Referencia (tu PDP)")
        ref_url = st.text_input("URL de tu PDP (opcional, para calcular gaps):", placeholder="https://tu-tienda.com/tu-producto")

        st.subheader("🔍 Productos de la competencia")
        urls_input = st.text_area(
            "URLs de productos competidores (una por línea)",
            height=160,
            placeholder="https://www.amazon.es/...\nhttps://www.ebay.es/...\nhttps://... ",
        )
        run_pdp = st.button("🚀 Analizar PDPs")

        if run_pdp:
            urls = []
            if ref_url and ref_url.strip():
                urls.append(ref_url.strip())
            urls += [u.strip() for u in urls_input.splitlines() if u.strip()]
            urls = list(dict.fromkeys(urls))  # unique

            st.session_state["pdp_data"] = []
            for i, u in enumerate(urls, 1):
                with st.spinner(f"Extrayendo ({i}/{len(urls)}): {u}"):
                    html = fetch_html(u, force_zenrows=force_zen, js_render=True, delay=delay)
                    if not html and ZENROWS_KEY:
                        # último intento: forzar zenrows
                        html = fetch_html(u, force_zenrows=True, js_render=True, delay=delay)
                    data = assemble_product(u, html) if html else {}
                    if data:
                        st.session_state["pdp_data"].append(data)
                    else:
                        st.warning(f"No se pudo extraer contenido de: {u}")

            if st.session_state.get("pdp_data"):
                df = pd.DataFrame(st.session_state["pdp_data"])
                st.success(f"✅ {len(df)} PDP(s) extraídos")
                st.dataframe(df[["title", "price", "domain", "url"]], use_container_width=True)

                # Radar de completitud rápido
                def completeness(d):
                    s = 0
                    s += 1 if d.get("title") else 0
                    s += 1 if d.get("description") else 0
                    s += 1 if d.get("price") else 0
                    s += min(len(d.get("features", [])) / 5, 1)
                    s += min(len(d.get("specifications", {})) / 5, 1)
                    s += min(len(d.get("images", [])) / 3, 1)
                    return s / 6 * 100

                comps = [completeness(d) for d in st.session_state["pdp_data"]]
                st.info(f"📊 Completitud media PDP: {sum(comps)/len(comps):.1f}%")

    # ---------------------------
    # TAB 2 — Shopping
    # ---------------------------
    with tab2:
        st.subheader("🛒 Mercado (Google Shopping)")
        query = st.text_input("¿Qué producto quieres analizar?", placeholder="Ej.: auriculares bluetooth deportivos")
        num_results = st.slider("Número de resultados", 5, 50, top_n, 5)
        run_shop = st.button("🔎 Buscar en Shopping", type="primary", disabled=not query)

        if run_shop and query:
            analyzer = GoogleShoppingAnalyzer(country=country)
            with st.spinner("Consultando Google Shopping..."):
                products, error = analyzer.search_products_free(query, num_results=num_results)
            if error:
                st.warning(f"⚠️ {error}")
            if products:
                st.session_state["shopping_products"] = products
                st.success(f"✅ {len(products)} resultados")
                # Vista rápida
                cols = st.columns([3, 1, 1, 1])
                with cols[0]:
                    st.caption("Título")
                with cols[1]:
                    st.caption("Precio")
                with cols[2]:
                    st.caption("Tienda")
                with cols[3]:
                    st.caption("Enlace")

                for i, p in enumerate(products[:20], 1):
                    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
                    c1.markdown(f"**{i}. {p.get('title','Sin título')[:90]}**")
                    c2.write(p.get("price", "N/A"))
                    c3.write(p.get("source", "—"))
                    link = p.get("link", "#")
                    if link and link != "#":
                        c4.markdown(f"[Abrir]({link})")

                # Análisis
                analysis = analyzer.analyze_shopping_data(products)
                if analysis.get("sources"):
                    src_df = pd.DataFrame(list(analysis["sources"].items()), columns=["Tienda", "Productos"]).sort_values("Productos", ascending=False)
                    fig = px.bar(src_df, x="Productos", y="Tienda", orientation="h", title="Distribución por tienda", color="Productos")
                    fig.update_layout(height=380, yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig, use_container_width=True)

                if analysis.get("price_ranges"):
                    pr = analysis["price_ranges"]
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("💵 Mín", f"{pr['min']:.2f}€")
                    m2.metric("💸 Máx", f"{pr['max']:.2f}€")
                    m3.metric("📊 Prom", f"{pr['avg']:.2f}€")
                    m4.metric("➗ Mediana", f"{pr['median']:.2f}€")

    # ---------------------------
    # TAB 3 — Matching & Insights
    # ---------------------------
    with tab3:
        st.subheader("🔗 Emparejamiento PDP ↔ Shopping y KPIs")
        pdps = st.session_state.get("pdp_data", [])
        shop = st.session_state.get("shopping_products", [])
        if not pdps:
            st.info("Añade PDPs en la pestaña **PDP por URLs**.")
        if not shop:
            st.info("Haz una búsqueda en **Google Shopping**.")
        if pdps and shop:
            threshold = st.slider("Umbral de similitud (score)", 50, 95, 70, 1)
            rows = []
            for d in pdps:
                best, score = match_pdp_to_shopping(d, shop, threshold=threshold)
                # Price Index
                my_amt, _ = parse_price_text(d.get("price", ""))
                mk_amt, _ = parse_price_text((best or {}).get("price", ""))
                price_idx = (my_amt / mk_amt) if (my_amt and mk_amt and mk_amt > 0) else None
                rows.append(
                    {
                        "pdp_title": d.get("title"),
                        "pdp_domain": d.get("domain"),
                        "pdp_price": d.get("price"),
                        "shop_title": (best or {}).get("title"),
                        "shop_price": (best or {}).get("price"),
                        "shop_source": (best or {}).get("source"),
                        "match_score": score,
                        "price_index": round(price_idx, 3) if price_idx else None,
                        "pdp_url": d.get("url"),
                        "shop_link": (best or {}).get("link"),
                    }
                )
            match_df = pd.DataFrame(rows)
            st.dataframe(match_df, use_container_width=True, hide_index=True)

            # KPI globales
            k1, k2, k3 = st.columns(3)
            valid_pi = match_df["price_index"].dropna()
            avg_pi = float(valid_pi.mean()) if not valid_pi.empty else None
            k1.metric("🎯 Coincidencias", f"{(match_df['match_score'] >= threshold).sum()} / {len(match_df)}")
            k2.metric("💶 Price Index medio", f"{avg_pi:.2f}" if avg_pi else "—")
            k3.metric("📈 Score medio", f"{match_df['match_score'].mean():.0f}")

            # Gráficos
            have_prices = match_df.dropna(subset=["price_index"])
            if not have_prices.empty:
                fig = px.scatter(
                    have_prices,
                    x="match_score",
                    y="price_index",
                    hover_data=["pdp_title", "shop_title", "shop_source"],
                    title="Matching Score vs Price Index",
                )
                st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # TAB 4 — Exportar
    # ---------------------------
    with tab4:
        st.subheader("📤 Exportaciones")
        if st.session_state.get("pdp_data"):
            df = pd.DataFrame(st.session_state["pdp_data"])
            st.download_button(
                "⬇️ Descargar PDPs (CSV)",
                df.to_csv(index=False).encode("utf-8"),
                file_name=f"pdp_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        if st.session_state.get("shopping_products"):
            sdfs = pd.DataFrame(st.session_state["shopping_products"])
            st.download_button(
                "⬇️ Descargar Shopping (CSV)",
                sdfs.to_csv(index=False).encode("utf-8"),
                file_name=f"shopping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.markdown("---")
    st.caption("v3.0 • Zenrows fallback • Structured Data opcional • Matching & Insights")


if __name__ == "__main__":
    main()
