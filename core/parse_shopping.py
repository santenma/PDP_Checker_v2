from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urlparse

from bs4 import BeautifulSoup

from core.http import fetch_html
from core.parse_pdp import clean_text, parse_price_text


# ---------------------------
# Búsqueda (Shopping + fallback Web)
# ---------------------------
def search_shopping(
    query: str,
    country: str = "es",
    num_results: int = 20,
    force_zenrows: bool = False,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Devuelve (productos, error). Intenta vertical Shopping; si hay pocos resultados, usa Web.
    Cada producto: {title, price, source, link}
    """
    if not query or not query.strip():
        return [], "Query vacío"

    errors: List[str] = []
    products: List[Dict[str, Any]] = []

    q = quote_plus(query.strip())
    shopping_url = f"https://www.google.com/search?tbm=shop&hl={country}&q={q}"

    # 1) Intento en Shopping
    html = fetch_html(shopping_url, force_zenrows=force_zenrows, js_render=True)
    if not html:
        # forzamos Zenrows si no se forzó antes
        html = fetch_html(shopping_url, force_zenrows=True, js_render=True)
    if html:
        products.extend(parse_shopping_html(html))
    else:
        errors.append("Shopping: sin respuesta")

    # 2) Fallback a Web si hay pocos
    if len(products) < 3:
        web_url = f"https://www.google.com/search?hl={country}&q={q}"
        html2 = fetch_html(web_url, force_zenrows=False, js_render=False)
        if not html2:
            html2 = fetch_html(web_url, force_zenrows=True, js_render=False)
        if html2:
            products.extend(parse_web_html(html2))
        else:
            errors.append("Web: sin respuesta")

    # Unificar y truncar
    uniq: List[Dict[str, Any]] = []
    seen = set()
    for p in products:
        k = (p.get("title", ""), p.get("source", ""), p.get("link", ""))
        if k not in seen and p.get("title"):
            seen.add(k)
            uniq.append(p)

    return uniq[: num_results], ("; ".join(errors) if errors else None)


# ---------------------------
# Parsing de tarjetas Shopping
# ---------------------------
def parse_shopping_html(html: str) -> List[Dict[str, Any]]:
    """
    Extrae tarjetas de Google Shopping desde HTML.
    Campos: title, price, source, link
    """
    out: List[Dict[str, Any]] = []
    if not html:
        return out

    soup = BeautifulSoup(html, "html.parser")

    # Contenedores típicos (rotan con frecuencia; listamos varios)
    cards = soup.select(
        "div.sh-dgr__content, "
        "div.pslires, "
        "div.sh-pr__product-results, "
        'div[data-docid][data-offer-id], '
        "div.sh-dlr__list-result, "
        "div.sh-dlr__content"
    )
    if not cards:
        # fallback: algunos layouts agrupan en 'div.sh-pr__product-results > div'
        container = soup.select_one("div.sh-pr__product-results")
        if container:
            cards = container.select("div")

    for card in cards:
        # Título
        title = ""
        t_candidates = [
            card.select_one("h3"),
            card.select_one("a[title]"),
            card.select_one("a[data-ttl]"),
        ]
        for t in t_candidates:
            if t and clean_text(t.get_text()):
                title = clean_text(t.get_text())
                break
        if not title:
            # fallback más agresivo
            txt = clean_text(card.get_text(" ", strip=True))
            title = txt[:140] if txt else ""

        # Precio
        price_txt = ""
        p_candidates = card.select(
            "span.a8Pemb, span.kHxwFf, span.OFFNJd, span.T14wmb, div.a8Pemb"
        )
        if not p_candidates:
            # fallback: cualquier texto con patrón de precio
            for el in card.find_all(["span", "div"]):
                t = clean_text(el.get_text())
                if _looks_like_price(t):
                    price_txt = t
                    break
        else:
            price_txt = clean_text(p_candidates[0].get_text())

        # Tienda / fuente
        seller = ""
        s_candidates = card.select("div.aULzUe, div.aEZQsc, div.E5ocAb")
        if s_candidates:
            seller = clean_text(s_candidates[0].get_text())

        # Enlace
        link = "#"
        a = card.select_one("a[href]")
        if a and a.get("href"):
            href = a["href"]
            if href.startswith("/"):
                link = "https://www.google.com" + href
            else:
                link = href

        # Normalizar precio
        amount, _cur = parse_price_text(price_txt)
        price = f"{amount:.2f} €" if amount else (price_txt or "N/A")

        if title:
            out.append(
                {
                    "title": title,
                    "price": price,
                    "source": seller or "Google Shopping",
                    "link": link,
                }
            )

    return out


# ---------------------------
# Parsing de resultados Web (fallback)
# ---------------------------
def parse_web_html(html: str) -> List[Dict[str, Any]]:
    """
    Extrae resultados orgánicos como fallback.
    Campos: title, price (N/A), source, link, description (si existe)
    """
    out: List[Dict[str, Any]] = []
    if not html:
        return out

    soup = BeautifulSoup(html, "html.parser")
    blocks = soup.select("div.g, div.MjjYud, div.tF2Cxc, div.NJo7tc")

    for b in blocks:
        a = b.select_one("a[href]")
        h3 = b.select_one("h3")
        snip = b.select_one("div.VwiC3b, div.Uroaid, div.ynAwRc")
        if a and h3:
            link = a.get("href", "#")
            src = urlparse(link).netloc
            out.append(
                {
                    "title": clean_text(h3.get_text()),
                    "price": "N/A",
                    "source": src or "web",
                    "link": link,
                    "description": clean_text(snip.get_text()) if snip else "",
                }
            )
    return out


# ---------------------------
# Análisis básico
# ---------------------------
def analyze_shopping_data(products: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calcula métricas básicas sobre una lista de productos Shopping.
    """
    from collections import Counter

    if not products:
        return {
            "total_products": 0,
            "sources": {},
            "price_stats": None,
            "common_terms": Counter(),
            "has_data": False,
        }

    # Fuentes
    sources: Dict[str, int] = {}
    for p in products:
        src = p.get("source", "desconocido")
        sources[src] = sources.get(src, 0) + 1

    # Precios
    prices: List[float] = []
    for p in products:
        amt, _cur = parse_price_text(p.get("price", ""))
        if amt and 0.01 < amt < 100000:
            prices.append(amt)
    price_stats = None
    if prices:
        prices_sorted = sorted(prices)
        n = len(prices_sorted)
        median = prices_sorted[n // 2] if n % 2 else (prices_sorted[n // 2 - 1] + prices_sorted[n // 2]) / 2
        price_stats = {
            "min": min(prices),
            "max": max(prices),
            "avg": sum(prices) / len(prices),
            "median": median,
            "count": len(prices),
        }

    # Términos comunes de títulos/descripciones
    text = " ".join([f"{p.get('title','')} {p.get('description','')}" for p in products])
    words = re.findall(r"\b[a-záéíóúñü0-9]{3,}\b", text.lower())
    stop = {
        "para", "con", "por", "del", "las", "los", "una", "uno", "desde",
        "hasta", "más", "muy", "todo", "todos", "este", "esta", "estos",
        "estas", "ese", "esa", "esos", "esas", "the", "and", "for", "with",
    }
    filtered = [w for w in words if w not in stop]
    common_terms = Counter(filtered)

    return {
        "total_products": len(products),
        "sources": sources,
        "price_stats": price_stats,
        "common_terms": common_terms,
        "has_data": True,
    }


# ---------------------------
# Utils
# ---------------------------
_PRICE_HINT = re.compile(r"(\d{1,3}(\.\d{3})*|\d+)([.,]\d{2})?\s?(€|eur|usd|gbp|\$|£)", re.IGNORECASE)

def _looks_like_price(text: str) -> bool:
    if not text or len(text) > 60:
        return False
    return bool(_PRICE_HINT.search(text))


__all__ = [
    "search_shopping",
    "parse_shopping_html",
    "parse_web_html",
    "analyze_shopping_data",
]
