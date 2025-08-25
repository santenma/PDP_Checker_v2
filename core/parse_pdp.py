from __future__ import annotations
import re
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup

# Dependencias opcionales (el módulo funciona con fallbacks si no están instaladas)
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


# ---------------------------
# Helpers
# ---------------------------
def clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()


def parse_price_text(text: Optional[str]):
    """
    Devuelve (amount_float, currency | None).
    Si existe price-parser, lo usa; si no, hace fallback simple por regex.
    """
    if not text:
        return None, None
    if PRICE_PARSER_OK:
        p = Price.fromstring(text)
        return p.amount_float, (p.currency or None)
    # Fallback naive
    t = text.replace(",", ".")
    m = re.search(r"(\d+(?:\.\d{2})?)", t)
    return (float(m.group(1)) if m else None), None


# ---------------------------
# 1) Schema.org Product
# ---------------------------
def _find_product_in_jsonld(node: Any) -> Optional[Dict[str, Any]]:
    """
    Busca objetos @type Product en estructuras JSON-LD arbitrarias (incluye @graph).
    """
    if isinstance(node, dict):
        t = node.get("@type")
        if t == "Product" or (isinstance(t, list) and "Product" in t):
            return node
        # @graph o propiedades anidadas
        for k, v in node.items():
            found = _find_product_in_jsonld(v)
            if found:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_product_in_jsonld(item)
            if found:
                return found
    return None


def parse_structured_product(html: str, url: str) -> Dict[str, Any]:
    """
    Intenta extraer un objeto Product desde JSON-LD/Microdata.
    Si no encuentra, devuelve {}.
    """
    if not EXSTRUCT_OK or not html:
        return {}

    try:
        base_url = get_base_url(html, url)
        data = extruct.extract(
            html,
            base_url=base_url,
            syntaxes=["json-ld", "microdata", "opengraph"],
        )

        # JSON-LD (más confiable)
        for ld in data.get("json-ld", []):
            if isinstance(ld, dict):
                found = _find_product_in_jsonld(ld)
                if found:
                    return found

        # Microdata (properties)
        for md in data.get("microdata", []):
            types = md.get("type", []) or []
            if any("Product" in t for t in types):
                props = md.get("properties", {}) or {}
                if props:
                    return props

        # Como ayuda, devolvemos OG si no hay Product (mejor que vacío)
        # *El OG detallado lo maneja parse_og()*
        return {}
    except Exception:
        return {}


# ---------------------------
# 2) Open Graph
# ---------------------------
def parse_og(soup: BeautifulSoup) -> Dict[str, str]:
    """
    Extrae Open Graph básico como dict: {'og:title': ..., 'og:description': ...}
    """
    out: Dict[str, str] = {}
    for tag in soup.select("meta[property^='og:'], meta[name^='og:']"):
        prop = tag.get("property") or tag.get("name")
        content = tag.get("content")
        if prop and content:
            out[prop] = content
    return out


# ---------------------------
# 3) Heurísticos
# ---------------------------
def extract_heuristics(soup: BeautifulSoup) -> Dict[str, Any]:
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

    # Features
    features: List[str] = []
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
    # Unique / case-insensitive
    seen = set()
    features = [f for f in features if not (f.lower() in seen or seen.add(f.lower()))][:50]

    # Especificaciones (tablas y listas descriptivas)
    specs: Dict[str, str] = {}
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

    # Breadcrumbs / categorías
    categories: List[str] = []
    for sel in ["nav.breadcrumb a", ".breadcrumb a", "[itemprop='itemListElement'] a", "a[rel='breadcrumb']"]:
        for a in soup.select(sel):
            txt = clean_text(a.get_text())
            if txt and txt.lower() not in ("home", "inicio"):
                categories.append(txt)
    categories = list(dict.fromkeys(categories))[:10]

    # Filtros (heurístico)
    filters_: List[str] = []
    for sel in ["[class*='filter'] a", "[class*='facet'] a", "[class*='filter'] label", "[class*='facet'] label"]:
        for el in soup.select(sel):
            txt = clean_text(el.get_text())
            if txt and 2 < len(txt) < 40:
                filters_.append(txt)
    filters_ = list(dict.fromkeys(filters_))[:40]

    # Imágenes
    images: List[str] = []
    for img in soup.select("img"):
        src = img.get("src") or img.get("data-src") or img.get("content")
        if src and src.startswith("http"):
            images.append(src)
    images = list(dict.fromkeys(images))[:10]

    # Precio (heurístico)
    price_text = ""
    for el in soup.select(
        "[class*='price'], [class*='cost'], [class*='amount'], "
        "[id*='price'], span[itemprop='price'], meta[itemprop='price']"
    ):
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


# ---------------------------
# Ensamblado final
# ---------------------------
def assemble_product(url: str, html: str) -> Dict[str, Any]:
    """
    Devuelve un dict con los campos normalizados de producto.
    Prioridad: structured > OG > heurísticos.
    """
    if not html:
        return {}

    soup = BeautifulSoup(html, "html.parser")
    structured = parse_structured_product(html, url) or {}
    og = parse_og(soup)
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
        heur.get("title"),
    )
    description = pick(
        structured.get("description"),
        og.get("og:description"),
        heur.get("description"),
    )

    # Precio: candidato desde structured.offers (dict o lista), heurístico y OG
    candidate_prices: List[str] = []
    offers = structured.get("offers")
    if isinstance(offers, dict):
        if offers.get("price"):
            candidate_prices.append(str(offers.get("price")))
        # Algunas implementaciones usan priceSpecification
        ps = offers.get("priceSpecification", {})
        if isinstance(ps, dict) and ps.get("price"):
            candidate_prices.append(str(ps.get("price")))
    elif isinstance(offers, list):
        for off in offers:
            if isinstance(off, dict) and off.get("price"):
                candidate_prices.append(str(off.get("price")))
                break  # primera válida

    if heur.get("raw_price"):
        candidate_prices.append(heur["raw_price"])
    if og.get("product:price:amount"):
        candidate_prices.append(og["product:price:amount"])

    price_val, currency = None, None
    for prw in candidate_prices:
        price_val, currency = parse_price_text(prw)
        if price_val:
            break

    # Brand puede venir como string o dict {'name': ...}
    brand = structured.get("brand")
    if isinstance(brand, dict):
        brand = brand.get("name")

    # Rating / reviews (aggregateRating puede ser dict o lista)
    rating_val, review_count = None, None
    ar = structured.get("aggregateRating")
    if isinstance(ar, dict):
        rating_val = ar.get("ratingValue")
        review_count = ar.get("reviewCount") or ar.get("ratingCount")

    # Availability / seller (offers dict|list)
    availability, seller = None, None
    def _extract_offer_fields(offer):
        av = offer.get("availability")
        se = offer.get("seller")
        if isinstance(se, dict):
            se = se.get("name") or json.dumps(se)
        return av, se

    if isinstance(offers, dict):
        availability, seller = _extract_offer_fields(offers)
    elif isinstance(offers, list) and offers:
        availability, seller = _extract_offer_fields(offers[0])

    product_data: Dict[str, Any] = {
        "url": url,
        "title": title,
        "description": description,
        "features": heur.get("features", []),
        "specifications": heur.get("specifications", {}),
        "categories": heur.get("categories", []),
        "filters": heur.get("filters", []),
        "images": heur.get("images", []),
        "price": (f"{price_val:.2f} {currency}" if price_val else (heur.get("raw_price") or "")),
        "brand": brand,
        "sku": structured.get("sku"),
        "mpn": structured.get("mpn"),
        "gtin": structured.get("gtin13") or structured.get("gtin") or structured.get("gtin8") or structured.get("gtin12"),
        "rating": rating_val,
        "reviewCount": review_count,
        "availability": availability,
        "seller": seller,
        "extracted_at": datetime.now().isoformat(),
    }
    return product_data


__all__ = [
    "clean_text",
    "parse_price_text",
    "parse_structured_product",
    "parse_og",
    "extract_heuristics",
    "assemble_product",
]
