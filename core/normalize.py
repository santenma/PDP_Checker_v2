from __future__ import annotations
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

# Dependencias opcionales
try:
    from price_parser import Price
    _PRICE_PARSER = True
except Exception:
    _PRICE_PARSER = False

try:
    from babel.numbers import get_currency_symbol
    _BABEL = True
except Exception:
    _BABEL = False

# Reutilizamos el parser de precio simple como respaldo
from core.parse_pdp import parse_price_text as _fallback_parse_price


# =========================
# Helpers de texto y números
# =========================
def strip_accents(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def norm_key(s: str) -> str:
    s = strip_accents(s or "")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def to_float(s: str) -> Optional[float]:
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None
    # Cambia coma decimal por punto
    t = t.replace(",", ".")
    m = re.search(r"[-+]?\d+(?:\.\d+)?", t)
    try:
        return float(m.group(0)) if m else None
    except Exception:
        return None


# =========================
# Moneda y precio
# =========================
# Mapa simple símbolo → código
SYMBOL_TO_CODE = {
    "€": "EUR", "$": "USD", "£": "GBP", "¥": "JPY", "₩": "KRW", "₹": "INR",
    "C$": "CAD", "A$": "AUD", "₽": "RUB", "R$": "BRL", "₺": "TRY",
}

# Tasas de ejemplo (no actualizadas). Reemplaza en runtime con `rates` si quieres exactitud.
EXAMPLE_RATES_TO_EUR = {
    "EUR": 1.0,
    "USD": 0.92,
    "GBP": 1.18,
    "JPY": 0.0062,
    "CAD": 0.67,
    "AUD": 0.61,
    "BRL": 0.18,
    "MXN": 0.05,
}

_CURRENCY_CODE_HINT = re.compile(r"\b(EUR|USD|GBP|JPY|CAD|AUD|BRL|MXN)\b", re.IGNORECASE)

def detect_currency_code(text: str, default: str = "EUR") -> str:
    """Detecta el código de moneda a partir de un texto de precio."""
    if not text:
        return default
    # Símbolo directo
    for sym, code in SYMBOL_TO_CODE.items():
        if sym in text:
            return code
    # Código textual
    m = _CURRENCY_CODE_HINT.search(text)
    if m:
        return m.group(1).upper()
    # '€', '$', '£' sin prefijo especial
    if "€" in text:
        return "EUR"
    if "$" in text:
        # Si aparece USD explícito ya habría hecho match. Asumimos USD por defecto.
        return "USD"
    if "£" in text:
        return "GBP"
    return default

def parse_price(text: str, default_currency: str = "EUR") -> Tuple[Optional[float], Optional[str]]:
    """
    Devuelve (importe, currency_code). Usa price-parser si está disponible; si no, fallback local.
    """
    if not text:
        return None, None
    if _PRICE_PARSER:
        p = Price.fromstring(text)
        amount = p.amount_float
        currency = p.currency or detect_currency_code(text, default_currency)
        return amount, currency
    # Fallback
    amount, _ = _fallback_parse_price(text)
    code = detect_currency_code(text, default_currency)
    return amount, code

def convert_currency(amount: Optional[float], from_code: Optional[str], to_code: str = "EUR",
                     rates: Optional[Dict[str, float]] = None) -> Optional[float]:
    """
    Convierte amount desde from_code → to_code usando `rates` (mapa a EUR o al target).
    Si no hay tasas, devuelve amount si ya está en to_code o None si no se puede convertir.
    """
    if amount is None or from_code is None:
        return None
    from_code = from_code.upper()
    to_code = to_code.upper()
    if from_code == to_code:
        return float(amount)

    # Si nos pasan un map de tasas a EUR, normalizamos
    rate_map = rates or EXAMPLE_RATES_TO_EUR
    # Si el mapa está definido '→EUR'
    if to_code == "EUR":
        r = rate_map.get(from_code)
        return float(amount) * r if r else None
    else:
        # Convertir de 'from' a EUR, y EUR a 'to' si tenemos ambas
        r_from = rate_map.get(from_code)
        r_to = rate_map.get(to_code)
        if r_from and r_to:
            eur = float(amount) * r_from
            return eur / r_to
        return None

def normalize_price_field(price_text: str,
                          target_currency: str = "EUR",
                          rates: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Normaliza un campo de precio textual a una estructura:
    {
        'amount_original': float|None,
        'currency_original': 'USD'|'EUR'|...|None,
        'amount': float|None,            # convertido a target_currency
        'currency': 'EUR'                # igual a target_currency si se pudo convertir
    }
    """
    amt, cur = parse_price(price_text or "")
    amount_norm = convert_currency(amt, cur, to_code=target_currency, rates=rates) if amt is not None else None
    out = {
        "amount_original": amt,
        "currency_original": cur,
        "amount": amount_norm if amount_norm is not None else amt if cur == target_currency else None,
        "currency": target_currency if (amount_norm is not None or (cur == target_currency and amt is not None)) else cur,
    }
    return out


# =========================
# Canonización de marca/modelo
# =========================
def canonicalize_brand(brand: Optional[str], aliases: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    Canoniza marca por normalización + alias (p.ej., 'hp'→'HP', 'hewlett-packard'→'HP').
    """
    if not brand:
        return None
    b = norm_key(brand)
    aliases = aliases or {}
    # Busca exacto en alias normalizado
    for k, v in aliases.items():
        if norm_key(k) == b:
            return v
    # Heurística: mayúsculas preservando acrónimos cortos
    out = strip_accents(brand).strip()
    if len(out) <= 4:
        return out.upper()
    # Capitalizar palabra a palabra
    return " ".join(w.upper() if len(w) <= 3 else w.capitalize() for w in re.split(r"\s+", out))


MODEL_TOKEN = re.compile(r"[A-Z0-9][A-Z0-9\-_/\.]{1,}", re.IGNORECASE)
def canonicalize_model(text: Optional[str], brand: Optional[str] = None) -> Optional[str]:
    """
    Extrae un token de modelo razonable del título/descripcion (heurístico).
    Elimina la marca si está presente.
    """
    if not text:
        return None
    t = strip_accents(text)
    if brand:
        t = re.sub(re.escape(strip_accents(brand)), " ", t, flags=re.IGNORECASE)
    # Busca tokens tipo "SM-G991B", "XR500", "1234-AB"
    m = MODEL_TOKEN.search(t)
    return m.group(0).upper() if m else None


# =========================
# Unidades: almacenamiento, peso, dimensiones, pantalla, batería
# =========================
_STORAGE = re.compile(r"(\d+(?:[\.,]\d+)?)\s*(tb|gb|mb|kb)", re.IGNORECASE)
_WEIGHT  = re.compile(r"(\d+(?:[\.,]\d+)?)\s*(kg|g|lb|lbs|oz)", re.IGNORECASE)
# Dimensiones en 'L x W x H' (cm/mm/in), separadores 'x', '×', '*'
_DIMS     = re.compile(r"(\d+(?:[\.,]\d+)?)\s*(mm|cm|in|inch|\")?\s*[x×*]\s*(\d+(?:[\.,]\d+)?)\s*(mm|cm|in|inch|\")?(?:\s*[x×*]\s*(\d+(?:[\.,]\d+)?)\s*(mm|cm|in|inch|\")?)?", re.IGNORECASE)
_SCREEN   = re.compile(r"(\d+(?:[\.,]\d+)?)\s*(in|inch|\"|cm)", re.IGNORECASE)
_BATTERY  = re.compile(r"(\d{3,5})\s*(mah)", re.IGNORECASE)

def _u_float(s: str) -> float:
    return float(s.replace(",", "."))

def _to_gb(v: float, unit: str) -> float:
    u = unit.lower()
    if u == "tb":
        return v * 1024.0
    if u == "gb":
        return v
    if u == "mb":
        return v / 1024.0
    if u == "kb":
        return v / (1024.0 * 1024.0)
    return v

def normalize_storage(text: str) -> Optional[float]:
    """
    Devuelve capacidad en GB (float), o None si no detecta.
    """
    if not text:
        return None
    m = _STORAGE.search(text)
    if not m:
        return None
    v = _u_float(m.group(1))
    unit = m.group(2)
    return round(_to_gb(v, unit), 3)

def _to_grams(v: float, unit: str) -> float:
    u = unit.lower()
    if u == "kg":
        return v * 1000.0
    if u == "g":
        return v
    if u in ("lb", "lbs"):
        return v * 453.59237
    if u == "oz":
        return v * 28.349523125
    return v

def normalize_weight(text: str) -> Optional[float]:
    """
    Devuelve peso en gramos (float).
    """
    if not text:
        return None
    m = _WEIGHT.search(text)
    if not m:
        return None
    v = _u_float(m.group(1))
    u = m.group(2)
    return round(_to_grams(v, u), 2)

def _to_mm(v: float, unit: Optional[str]) -> float:
    if not unit:
        return v  # asumimos mm si no hay unidad
    u = unit.lower()
    if u == "mm":
        return v
    if u == "cm":
        return v * 10.0
    if u in ("in", "inch", '"'):
        return v * 25.4
    return v

def normalize_dimensions(text: str) -> Optional[Dict[str, float]]:
    """
    Devuelve dict {'l_mm':..,'w_mm':..,'h_mm':..} si detecta patrón LxWxH.
    """
    if not text:
        return None
    m = _DIMS.search(text)
    if not m:
        return None
    l, lu, w, wu, h, hu = m.groups()
    out = {
        "l_mm": round(_to_mm(_u_float(l), lu), 2),
        "w_mm": round(_to_mm(_u_float(w), wu), 2),
    }
    if h:
        out["h_mm"] = round(_to_mm(_u_float(h), hu), 2)
    return out

def normalize_screen_size(text: str) -> Optional[Dict[str, float]]:
    """
    Devuelve {'size_in': float, 'size_cm': float} si detecta tamaño de pantalla.
    """
    if not text:
        return None
    m = _SCREEN.search(text)
    if not m:
        return None
    v = _u_float(m.group(1))
    u = m.group(2).lower()
    if u in ('cm',):
        size_in = v / 2.54
        size_cm = v
    else:
        size_in = v
        size_cm = v * 2.54
    return {"size_in": round(size_in, 2), "size_cm": round(size_cm, 2)}

def normalize_battery(text: str) -> Optional[int]:
    """
    Devuelve capacidad de batería en mAh.
    """
    if not text:
        return None
    m = _BATTERY.search(text.replace(".", ""))
    if not m:
        return None
    return int(m.group(1))


# =========================
# Canonicalización de claves de specs
# =========================
# Reglas (orden importa). Añade/edita según tu vertical.
CANON_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^brand|^marca$"), "brand"),
    (re.compile(r"^model|^modelo$"), "model"),
    (re.compile(r"^sku$"), "sku"),
    (re.compile(r"^mpn$"), "mpn"),
    (re.compile(r"^gtin|ean|upc$"), "gtin"),

    (re.compile(r"^price|^precio$"), "price"),
    (re.compile(r"^availability|^disponibilidad$"), "availability"),

    (re.compile(r"^weight|^peso$"), "weight_g"),
    (re.compile(r"^dimensions?|^dimensiones?$"), "dimensions_mm"),
    (re.compile(r"^width|^ancho$"), "width_mm"),
    (re.compile(r"^height|^alto$"), "height_mm"),
    (re.compile(r"^depth|^fondo|^profundidad$"), "depth_mm"),

    (re.compile(r"^storage|^almacenamiento|^capacidad$"), "storage_gb"),
    (re.compile(r"^battery|bateria|^capacidad de bateria$"), "battery_mah"),
    (re.compile(r"^screen|^pantalla|^display$"), "screen"),

    (re.compile(r"^color$"), "color"),
    (re.compile(r"^material$"), "material"),
]

def canonical_key(k: str) -> str:
    key = norm_key(k)
    for pat, out in CANON_RULES:
        if pat.search(key):
            return out
    # snake_case por defecto
    key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return key


# =========================
# Normalización de specs (k,v)
# =========================
def normalize_spec_kv(k: str, v: Any) -> Tuple[str, Any, Dict[str, Any]]:
    """
    Devuelve (canonical_key, canonical_value, extras_dict)
    - canonical_value: preferentemente numérico/estructurado si aplica
    - extras_dict: campos derivados (ej. l_mm/w_mm/h_mm)
    """
    ck = canonical_key(k)
    text_v = str(v) if v is not None else ""

    extras: Dict[str, Any] = {}

    if ck == "weight_g":
        w = normalize_weight(text_v)
        return ck, w if w is not None else v, extras

    if ck in ("dimensions_mm", "width_mm", "height_mm", "depth_mm"):
        dims = normalize_dimensions(text_v)
        if dims:
            # Si venía como 'dimensions', devolvemos dict completo
            if ck == "dimensions_mm":
                return ck, dims, extras
            # Si era ancho/alto/fondo suelto
            map_single = {
                "width_mm": ("l_mm", "w_mm"),   # no sabemos si W o L, tomamos el mayor como ancho si hiciera falta
                "height_mm": ("h_mm",),
                "depth_mm": ("w_mm", "l_mm"),
            }
            # Mejor devolver el valor más probable si existe
            candidates = map_single.get(ck, ())
            for c in candidates:
                if c in dims:
                    return ck, dims[c], extras
            return ck, dims, extras
        return ck, v, extras

    if ck == "storage_gb":
        gb = normalize_storage(text_v)
        return ck, gb if gb is not None else v, extras

    if ck == "battery_mah":
        mah = normalize_battery(text_v)
        return ck, mah if mah is not None else v, extras

    if ck == "screen":
        sc = normalize_screen_size(text_v)
        return ck, sc if sc else v, extras

    if ck == "price":
        # No convertir aquí (porque ya puede venir normalizado a nivel de entidad)
        amt, cur = parse_price(text_v)
        return ck, {"amount": amt, "currency": cur, "raw": text_v}, extras

    # Por defecto, devuelve valor original
    return ck, v, extras


def normalize_specs(specs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normaliza un dict de especificaciones mezcladas (ES/EN).
    - Canoniza claves.
    - Convierte unidades donde aplique.
    - Devuelve un nuevo dict con claves canónicas.
    """
    out: Dict[str, Any] = {}
    for k, v in (specs or {}).items():
        ck, cv, extra = normalize_spec_kv(k, v)
        # Merge inteligente
        if ck in out and isinstance(out[ck], dict) and isinstance(cv, dict):
            out[ck] = {**out[ck], **cv}
        else:
            out[ck] = cv
        for ek, ev in extra.items():
            out[ek] = ev
    return out


__all__ = [
    "parse_price",
    "convert_currency",
    "normalize_price_field",
    "canonicalize_brand",
    "canonicalize_model",
    "normalize_storage",
    "normalize_weight",
    "normalize_dimensions",
    "normalize_screen_size",
    "normalize_battery",
    "canonical_key",
    "normalize_spec_kv",
    "normalize_specs",
]
