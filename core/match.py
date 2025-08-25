from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import re
import unicodedata

# Fuzzy opcional
try:
    from rapidfuzz import fuzz
    _RAPIDFUZZ = True
except Exception:
    _RAPIDFUZZ = False

from core.parse_pdp import clean_text, parse_price_text


# ---------------------------
# Data structures
# ---------------------------
@dataclass
class MatchResult:
    pdp_idx: int
    shop_idx: Optional[int]
    score: int
    reason: str
    price_index: Optional[float]
    price_delta_pct: Optional[float]


# ---------------------------
# Normalización básica
# ---------------------------
def _normalize(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    # Mantener alfanum y espacios
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _signature(item: Dict[str, Any], kind: str) -> str:
    """
    Crea una firma de texto para comparar:
    - PDP: brand + title
    - SHOP: (brand si existe) + title
    """
    brand = item.get("brand") or ""
    title = item.get("title") or ""
    sig = f"{brand} {title}".strip()
    if not sig:
        # Fallback si faltan campos
        sig = item.get("description") or item.get("name") or ""
    return _normalize(sig)


def _fuzzy_score(a: str, b: str) -> int:
    if not a or not b:
        return 0
    if _RAPIDFUZZ:
        # Combinación estable y rápida
        tset = fuzz.token_set_ratio(a, b)
        part = fuzz.partial_ratio(a, b)
        wr = getattr(fuzz, "WRatio", None)
        w = wr(a, b) if wr else tset
        # Ponderamos para robustez
        score = 0.5 * tset + 0.2 * part + 0.3 * w
        return int(round(score))
    # Fallback naive: Jaccard de tokens
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0
    inter = len(ta & tb)
    union = len(ta | tb)
    return int(round(100 * inter / union))


def _price_metrics(pdp_price: Optional[str], shop_price: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """
    Devuelve (price_index, delta_pct). price_index = pdp/shop.
    delta_pct = (pdp - shop)/shop.
    """
    pa, _ = parse_price_text(pdp_price or "")
    pb, _ = parse_price_text(shop_price or "")
    if not pa or not pb or pb <= 0:
        return None, None
    price_index = pa / pb
    delta_pct = (pa - pb) / pb
    return round(price_index, 4), round(delta_pct, 4)


# ---------------------------
# Matching por claves duras
# ---------------------------
def _hard_key_match(pdp: Dict[str, Any], shops: List[Dict[str, Any]]) -> Optional[int]:
    keys = ["gtin", "mpn", "sku"]
    for k in keys:
        v = pdp.get(k)
        if not v:
            continue
        for j, s in enumerate(shops):
            if s.get(k) and str(s[k]).strip() == str(v).strip():
                return j
    return None


# ---------------------------
# API pública
# ---------------------------
def match_one(
    pdp: Dict[str, Any],
    shops: List[Dict[str, Any]],
    threshold: int = 70,
    prefer_price_proximity: bool = True,
) -> Tuple[Optional[int], int, str]:
    """
    Devuelve (idx_shop, score, reason) para un PDP concreto.
    - idx_shop None si no alcanza threshold.
    - reason indica 'key:gtin/mpn/sku' o 'fuzzy:title+brand'.
    """
    if not shops:
        return None, 0, "no_candidates"

    # 1) Clave dura
    j = _hard_key_match(pdp, shops)
    if j is not None:
        return j, 100, "key_match"

    # 2) Fuzzy
    sig_a = _signature(pdp, "pdp")
    best_idx, best_score = None, -1

    # Pre-cálculo de precio PDP para tiebreaker
    pdp_amt, _ = parse_price_text(pdp.get("price", ""))

    for idx, s in enumerate(shops):
        sig_b = _signature(s, "shop")
        sc = _fuzzy_score(sig_a, sig_b)
        # Tiebreaker por precio si las puntuaciones son similares
        if prefer_price_proximity and sc >= best_score - 2 and pdp_amt:
            shop_amt, _ = parse_price_text(s.get("price", ""))
            if shop_amt:
                # penaliza grandes desviaciones: cuanto más cerca, mejor
                rel_diff = abs(pdp_amt - shop_amt) / max(shop_amt, 1e-9)
                bonus = max(0, int(10 - min(10, rel_diff * 50)))  # hasta +10 si casi iguales
                sc = min(100, sc + bonus)

        if sc > best_score:
            best_score = sc
            best_idx = idx

    if best_idx is None or best_score < threshold:
        return None, int(best_score if best_score >= 0 else 0), "no_match"

    return best_idx, int(best_score), "fuzzy_match"


def match_greedy(
    pdps: List[Dict[str, Any]],
    shops: List[Dict[str, Any]],
    threshold: int = 70,
    one_to_one: bool = True,
    prefer_price_proximity: bool = True,
) -> List[MatchResult]:
    """
    Empareja en modo greedy. Si `one_to_one` es True, no repite el mismo shop en dos PDP
    (elige el de mayor score).
    """
    results: List[MatchResult] = []
    taken: set = set()

    # Primer pase: key-match con score 100 (si one_to_one, los bloquea)
    for i, pdp in enumerate(pdps):
        j = _hard_key_match(pdp, shops)
        if j is not None and (not one_to_one or j not in taken):
            pi, dlt = _price_metrics(pdp.get("price"), shops[j].get("price"))
            results.append(MatchResult(i, j, 100, "key_match", pi, dlt))
            if one_to_one:
                taken.add(j)
        else:
            results.append(MatchResult(i, None, 0, "pending", None, None))

    # Segundo pase: fuzzy para los pendientes
    for r in results:
        if r.reason != "pending":
            continue
        i = r.pdp_idx
        pdp = pdps[i]

        # Si one_to_one, filtra candidatos no tomados
        candidates = [
            (idx, s) for idx, s in enumerate(shops)
            if (not one_to_one) or (idx not in taken)
        ]
        if not candidates:
            r.reason = "no_candidates"
            continue

        # Ejecuta match_one contra el subconjunto válido
        subset = [s for _, s in candidates]
        chosen_rel_idx, score, reason = match_one(
            pdp,
            subset,
            threshold=threshold,
            prefer_price_proximity=prefer_price_proximity,
        )
        if chosen_rel_idx is None:
            r.score = score
            r.reason = reason
            continue

        j = candidates[chosen_rel_idx][0]
        r.shop_idx = j
        r.score = score
        r.reason = reason
        r.price_index, r.price_delta_pct = _price_metrics(
            pdp.get("price"), shops[j].get("price")
        )
        if one_to_one:
            taken.add(j)

    # Limpia los que quedaron 'pending'
    for r in results:
        if r.reason == "pending":
            r.reason = "no_match"

    return results


__all__ = [
    "MatchResult",
    "match_one",
    "match_greedy",
]
