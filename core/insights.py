from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import math
import re
from collections import Counter, defaultdict

# pandas es opcional
try:
    import pandas as pd  # type: ignore
    _PD = True
except Exception:
    _PD = False

from core.normalize import (
    normalize_specs,
    normalize_price_field,
    canonicalize_brand,
    canonicalize_model,
    parse_price,
)
from core.parse_pdp import clean_text, parse_price_text
from core.match import MatchResult


# =========================
# Utilidades estadísticas
# =========================
def _stats(values: List[float]) -> Optional[Dict[str, float]]:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    s = sum(vals)
    avg = s / n
    med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    var = sum((x - avg) ** 2 for x in vals) / n if n > 1 else 0.0
    std = var ** 0.5
    cv = std / avg if avg else 0.0
    return {
        "count": float(n),
        "min": float(vals[0]),
        "max": float(vals[-1]),
        "avg": float(avg),
        "median": float(med),
        "std": float(std),
        "cv": float(cv),
    }


# =========================
# Perfil de mercado (Shopping)
# =========================
def market_profile(shopping_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    prices = []
    sources = Counter()
    for p in shopping_items:
        amt, _cur = parse_price_text(p.get("price", ""))
        if amt and 0.01 < amt < 1e6:
            prices.append(amt)
        src = p.get("source") or "desconocido"
        sources[src] += 1

    return {
        "price_stats": _stats(prices),
        "sources": dict(sources),
        "has_data": bool(shopping_items),
    }


# =========================
# Price Index y agregados
# =========================
def price_kpis_from_matches(
    pdps: List[Dict[str, Any]],
    shopping_items: List[Dict[str, Any]],
    matches: List[MatchResult],
    target_currency: str = "EUR",
    rates: Optional[Dict[str, float]] = None,
    outlier_band: float = 0.10,  # ±10%
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    for m in matches:
        pdp = pdps[m.pdp_idx]
        shop = shopping_items[m.shop_idx] if (m.shop_idx is not None and m.shop_idx < len(shopping_items)) else None

        # Normalizamos precios a moneda objetivo (mejor comparabilidad)
        pdp_price = normalize_price_field(pdp.get("price", ""), target_currency, rates)
        shop_price = normalize_price_field((shop or {}).get("price", ""), target_currency, rates)

        pi = None
        dlt = None
        if pdp_price["amount"] and shop_price["amount"] and shop_price["amount"] > 0:
            pi = pdp_price["amount"] / shop_price["amount"]
            dlt = (pdp_price["amount"] - shop_price["amount"]) / shop_price["amount"]

        rows.append(
            {
                "pdp_title": pdp.get("title"),
                "pdp_domain": pdp.get("domain"),
                "pdp_url": pdp.get("url"),
                "pdp_price_amount": pdp_price["amount"],
                "pdp_price_currency": pdp_price["currency"],
                "shop_title": (shop or {}).get("title"),
                "shop_source": (shop or {}).get("source"),
                "shop_link": (shop or {}).get("link"),
                "shop_price_amount": shop_price["amount"],
                "shop_price_currency": shop_price["currency"],
                "match_score": m.score,
                "match_reason": m.reason,
                "price_index": pi,
                "price_delta_pct": dlt,
            }
        )

    # Agregados
    pis = [r["price_index"] for r in rows if r["price_index"] is not None]
    scores = [r["match_score"] for r in rows if isinstance(r["match_score"], (int, float))]
    outliers_hi = [pi for pi in pis if pi and pi > 1 + outlier_band]
    outliers_lo = [pi for pi in pis if pi and pi < 1 - outlier_band]

    result = {
        "rows": (pd.DataFrame(rows) if _PD else rows),
        "match_rate": float(sum(1 for r in rows if r["shop_title"]) / max(1, len(rows))),
        "avg_match_score": float(sum(scores) / len(scores)) if scores else None,
        "price_index_stats": _stats(pis),
        "outliers": {
            "above": len(outliers_hi),
            "below": len(outliers_lo),
            "band": outlier_band,
        },
    }
    return result


# =========================
# Coverage de especificaciones (PDP vs competidores)
# =========================
# Pesos por campo canónico (ajusta a tu vertical)
SPEC_WEIGHTS = {
    "brand": 2.5,
    "model": 2.0,
    "gtin": 2.0,
    "mpn": 1.8,
    "sku": 1.5,
    "price": 2.0,
    "availability": 1.2,
    "weight_g": 1.0,
    "dimensions_mm": 1.0,
    "width_mm": 0.5,
    "height_mm": 0.5,
    "depth_mm": 0.5,
    "storage_gb": 1.5,
    "battery_mah": 1.3,
    "screen": 1.3,
    "color": 0.6,
    "material": 0.5,
}

def spec_coverage(
    reference_pdp: Dict[str, Any],
    competitor_pdps: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calcula cobertura ponderada de specs del PDP de referencia frente al 'mercado' (competidores).
    - Normaliza specs y calcula qué campos clave están presentes en el mercado y ausentes en el ref.
    """
    # Normaliza specs
    ref_specs = normalize_specs(reference_pdp.get("specifications") or {})
    comp_specs_list = [normalize_specs(p.get("specifications") or {}) for p in competitor_pdps]

    # Campos relevantes en el mercado (presentes en >= 20% de competidores)
    field_freq: Counter = Counter()
    n = max(1, len(comp_specs_list))
    for sp in comp_specs_list:
        for k, v in sp.items():
            if v not in (None, "", {}, []):
                field_freq[k] += 1
    market_keys = {k for k, c in field_freq.items() if c / n >= 0.2}

    # Cálculo de cobertura ponderada
    total_weight = sum(SPEC_WEIGHTS.get(k, 0.5) for k in market_keys) or 1.0
    have_weight = 0.0
    missing: List[Tuple[str, float]] = []
    present: List[str] = []

    for k in sorted(market_keys):
        w = SPEC_WEIGHTS.get(k, 0.5)
        if ref_specs.get(k) not in (None, "", {}, []):
            have_weight += w
            present.append(k)
        else:
            missing.append((k, w))

    coverage = have_weight / total_weight
    missing_sorted = sorted(missing, key=lambda x: x[1], reverse=True)

    return {
        "coverage_weighted": round(coverage, 4),
        "present_keys": present,
        "missing_keys": [{"key": k, "weight": w} for k, w in missing_sorted],
        "market_keys": sorted(list(market_keys)),
        "market_field_frequency": dict(field_freq),
        "ref_specs_normalized": ref_specs,
    }


# =========================
# Cobertura de variantes (color / capacidad)
# =========================
_COLOR_SET = {
    "negro","black","blanco","white","azul","blue","rojo","red","verde","green","gris","gray","silver","plateado",
    "dorado","gold","morado","purple","amarillo","yellow","rosa","pink","beige","marron","brown","naranja","orange"
}
_CAPACITY = re.compile(r"\b(\d{2,4})\s?(gb|tb)\b", re.IGNORECASE)

def _variants_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    if not text:
        return None, None
    low = text.lower()
    color = None
    for c in _COLOR_SET:
        if re.search(rf"\b{re.escape(c)}\b", low):
            color = c
            break
    cap = None
    m = _CAPACITY.search(low)
    if m:
        cap = f"{m.group(1)}{m.group(2).lower()}"
    return color, cap

def variant_coverage(
    reference_pdp: Dict[str, Any],
    competitor_pdps: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Extrae variantes (color/capacidad) desde títulos y specs.
    Calcula qué variantes existen en el 'mercado' y cuáles faltan en el PDP de referencia.
    """
    def variants_for(p: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        title = p.get("title") or ""
        color, cap = _variants_from_text(title)
        # Revisión en specs normalizados (almacenamiento)
        ns = normalize_specs(p.get("specifications") or {})
        if not cap and isinstance(ns.get("storage_gb"), (int, float)):
            cap_gb = int(round(float(ns["storage_gb"])))
            cap = f"{cap_gb}gb"
        # Color en specs si existiera
        if not color and isinstance(ns.get("color"), str):
            color = ns["color"].lower()
        return color, cap

    ref_color, ref_cap = variants_for(reference_pdp)
    ref_set = set(filter(None, [ref_color, ref_cap]))

    market: set = set()
    for c in competitor_pdps:
        col, cap = variants_for(c)
        for v in (col, cap):
            if v:
                market.add(v)

    missing = sorted(list(market - ref_set))
    present = sorted(list(ref_set & market))

    return {
        "present_variants": present,
        "missing_variants": missing,
        "market_variants": sorted(list(market)),
        "ref_color": ref_color,
        "ref_capacity": ref_cap,
    }


# =========================
# Informe integrado
# =========================
def build_insights_report(
    pdps: List[Dict[str, Any]],
    shopping_items: List[Dict[str, Any]],
    matches: List[MatchResult],
    reference_idx: int = 0,
    target_currency: str = "EUR",
    rates: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Devuelve un informe integral con:
      - KPIs de precio por match (y agregados)
      - Perfil de mercado (Shopping)
      - Cobertura de specs ponderada
      - Cobertura de variantes (color/capacidad)
    """
    if not pdps:
        return {"error": "no_pdps"}

    # Precio + matching
    price_section = price_kpis_from_matches(
        pdps=pdps,
        shopping_items=shopping_items,
        matches=matches,
        target_currency=target_currency,
        rates=rates,
    )

    # Perfil de mercado
    market_section = market_profile(shopping_items)

    # Coverage specs/variantes (ref vs competidores)
    ref = pdps[min(reference_idx, len(pdps)-1)]
    competitors = [p for i, p in enumerate(pdps) if i != min(reference_idx, len(pdps)-1)]
    spec_section = spec_coverage(ref, competitors) if competitors else {
        "coverage_weighted": None, "present_keys": [], "missing_keys": []
    }
    variant_section = variant_coverage(ref, competitors) if competitors else {
        "present_variants": [], "missing_variants": [], "market_variants": []
    }

    return {
        "price": price_section,
        "market": market_section,
        "specs": spec_section,
        "variants": variant_section,
    }
