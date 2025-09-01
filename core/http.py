# core/http.py
# v1.0 — HTTP utilities with Zenrows fallback, retries, and light rate limiting.

from __future__ import annotations
import os
import re
import time
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode, urlparse

import requests
from requests import Response
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------------------------------
# Logging
# ------------------------------------------------------
logger = logging.getLogger("core.http")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ------------------------------------------------------
# Config / Env
# ------------------------------------------------------
def _get_zenrows_key() -> Optional[str]:
    # Evita dependencia dura de Streamlit; si existe, léelo de secrets
    key = os.getenv("ZENROWS_API_KEY")
    if key:
        return key
    try:
        import streamlit as st  # type: ignore
        return st.secrets.get("ZENROWS_API_KEY")  # type: ignore
    except Exception:
        return None

ZENROWS_KEY = _get_zenrows_key()

DEFAULT_HEADERS_POOL = [
    # Rotación básica y segura
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
]

# Rate limiting global (muy simple)
_LAST_REQUEST_TS = 0.0

def _respect_rate_limit(min_interval_sec: float) -> None:
    global _LAST_REQUEST_TS
    now = time.time()
    elapsed = now - _LAST_REQUEST_TS
    if elapsed < min_interval_sec:
        time.sleep(min_interval_sec - elapsed)
    _LAST_REQUEST_TS = time.time()

# ------------------------------------------------------
# Helpers
# ------------------------------------------------------
_CAPTCHA_PATTERNS = [
    "captcha",
    "unusual traffic",
    "verify you are a human",
    "are you a robot",
    "bot detection",
    "temporarily unavailable due to",
    "access denied",
    "cloudflare",
    "please wait",
    "checking your browser",
    "ddos protection",
    "forbidden",
]

# Sitios conocidos con protección anti-bot fuerte
PROTECTED_SITES = {
    "pccomponentes.com": {
        "force_zenrows": True,
        "js_render": True,
        "premium_proxy": True,
        "antibot": True,
        "wait": 3000,  # Esperar 3s para que cargue JS
        "wait_for": ".product-name, h1, [itemprop='name']",  # Esperar elementos del producto
        "block_resources": "image,media,font",  # Bloquear recursos pesados
    },
    "mediamarkt.es": {
        "force_zenrows": True,
        "js_render": True,
        "wait": 2000,
    },
    # Añadir más sitios según necesidad
}

def looks_like_captcha(html: str) -> bool:
    if not html:
        return False
    low = html.lower()
    return any(p in low for p in _CAPTCHA_PATTERNS)

def _pick_ua() -> str:
    import random
    return random.choice(DEFAULT_HEADERS_POOL)

def _is_ok_status(status: int) -> bool:
    return 200 <= status < 300

def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"URL inválida: {url}")

+def _get_site_config(url: str) -> Dict[str, Any]:
    """
    Obtiene configuración específica para sitios con protección anti-bot
    """
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    
    # Buscar configuración exacta o con subdominios
    for site, config in PROTECTED_SITES.items():
        if site in domain or domain.endswith(f".{site}"):
            logger.info(f"Usando configuración especial para {site}")
            return config.copy()
    
    # Configuración por defecto
    return {}

# ------------------------------------------------------
# Options
# ------------------------------------------------------
@dataclass
class FetchOptions:
    timeout: int = 30
    retries: int = 2
    backoff_factor: float = 1.6
    min_interval_sec: float = 0.8  # rate limit suave
    force_zenrows: bool = False
    js_render: bool = True
    premium_proxy: bool = True
    antibot: bool = True
    headers: Optional[Dict[str, str]] = None  # si None, se genera con UA
    zenrows_params: Optional[Dict[str, str]] = None  # extras para Zenrows
    allow_redirects: bool = True
    wait: Optional[int] = None  # ms para esperar después de cargar
    wait_for: Optional[str] = None  # selector CSS para esperar

# ------------------------------------------------------
# Core requests
# ------------------------------------------------------
def _direct_request(url: str, opts: FetchOptions) -> Optional[Response]:
    headers = opts.headers.copy() if opts.headers else {}
    headers.setdefault("User-Agent", _pick_ua())
    headers.setdefault("Accept-Language", "es-ES,es;q=0.9,en;q=0.8")
    _respect_rate_limit(opts.min_interval_sec)
    try:
        r = requests.get(url, headers=headers, timeout=opts.timeout, allow_redirects=opts.allow_redirects)
        return r
    except requests.RequestException as e:
        logger.debug(f"Direct request error: {e}")
        return None

def _zenrows_request(url: str, opts: FetchOptions) -> Optional[Response]:
    if not ZENROWS_KEY:
        logger.debug("Zenrows no disponible (sin API key).")
        return None

    params = {"url": url}
    if opts.js_render:
        params["js_render"] = "true"
    if opts.premium_proxy:
        params["premium_proxy"] = "true"
    if opts.antibot:
        params["antibot"] = "true"
    
    # Parámetros adicionales para bypass
    if opts.wait:
        params["wait"] = str(opts.wait)
    if opts.wait_for:
        params["wait_for"] = opts.wait_for
    if opts.zenrows_params and "block_resources" in opts.zenrows_params:
        params["block_resources"] = opts.zenrows_params["block_resources"]
        
    if opts.zenrows_params:
        # Permite inyectar parámetros adicionales como 'block_resources', 'custom_headers', etc.
        params.update({k: str(v) for k, v in opts.zenrows_params.items()})

    headers = opts.headers.copy() if opts.headers else {}
    headers.setdefault("User-Agent", _pick_ua())
    headers.setdefault("Accept-Language", "es-ES,es;q=0.9,en;q=0.8")

    url_api = "https://api.zenrows.com/v1/?" + urlencode(params)
    _respect_rate_limit(opts.min_interval_sec)
    try:
        r = requests.get(url_api, headers=headers, auth=(ZENROWS_KEY, ""), timeout=opts.timeout, allow_redirects=opts.allow_redirects)
        return r
    except requests.RequestException as e:
        logger.debug(f"Zenrows request error: {e}")
        return None

def _need_fallback(resp: Optional[Response]) -> bool:
    if resp is None:
        return True
    if not _is_ok_status(resp.status_code):
        return True
    if looks_like_captcha(resp.text):
        return True
    return False

# ------------------------------------------------------
# Public API
# ------------------------------------------------------
def fetch_html(url: str,
               force_zenrows: bool = False,
               js_render: bool = True,
               timeout: int = 30,
               retries: int = 2,
               backoff_factor: float = 1.6,
               min_interval_sec: float = 0.8,
               headers: Optional[Dict[str, str]] = None,
               zenrows_params: Optional[Dict[str, str]] = None) -> str:
    """
    Devuelve HTML de `url`. Intenta petición directa y, si hay bloqueo/CAPTCHA/403,
    hace fallback a Zenrows (si hay API key). Reintenta con backoff.
    """
    _validate_url(url)
    
    # Obtener configuración específica del sitio
    site_config = _get_site_config(url)
    
    # Aplicar configuración del sitio si existe
    if site_config:
        force_zenrows = site_config.get("force_zenrows", force_zenrows)
        js_render = site_config.get("js_render", js_render)
        if not zenrows_params:
            zenrows_params = {}
        zenrows_params.update({k: v for k, v in site_config.items() 
                              if k not in ["force_zenrows", "js_render", "premium_proxy", "antibot"]})
                       
    opts = FetchOptions(
        timeout=timeout,
        retries=max(0, retries),
        backoff_factor=backoff_factor,
        min_interval_sec=min_interval_sec,
        force_zenrows=force_zenrows,
        js_render=js_render,
        headers=headers,
        zenrows_params=zenrows_params,
        wait=site_config.get("wait"),
        wait_for=site_config.get("wait_for"),
    )

    attempt = 0
    while True:
        attempt += 1
        # 1) Ruta preferida (directa o Zenrows forzado)
        if opts.force_zenrows:
            primary = _zenrows_request(url, opts)
            fallback = _direct_request(url, opts)
        else:
            primary = _direct_request(url, opts)
            fallback = _zenrows_request(url, opts)

        resp = primary
        if _need_fallback(resp):
            resp = fallback

        if resp and _is_ok_status(resp.status_code) and not looks_like_captcha(resp.text):
            return resp.text or ""

        if attempt > opts.retries:
            # Último intento; si hay respuesta, devuélvela aunque sea vacía
            if resp is not None and resp.text is not None:
                return resp.text

            # Log específico para debugging
            if resp:
                logger.warning(f"URL {url} devolvió status {resp.status_code}")
                if resp.status_code == 403:
                    logger.error(f"403 Forbidden en {url} - El sitio está bloqueando el acceso")
                elif resp.status_code == 503:
                    logger.error(f"503 Service Unavailable en {url} - Posible protección anti-bot")
            else:
                logger.error(f"No se pudo conectar con {url}")
            
            return ""

        # Backoff exponencial antes de reintentar
        sleep_for = (opts.backoff_factor ** attempt) * 0.5
        time.sleep(min(sleep_for, 6.0))

def fetch_json(url: str, **kwargs) -> Optional[Dict[str, Any]]:
    """Obtiene JSON desde `url` (GET). Usa el mismo mecanismo de fallback."""
    html = fetch_html(url, **kwargs)
    if not html:
        return None
    try:
        return json.loads(html)
    except json.JSONDecodeError:
        # A veces se devuelve HTML; intentamos detectar un <pre> con JSON
        try:
            # extracción naive
            start = html.find("{")
            end = html.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(html[start : end + 1])
        except Exception:
            pass
        return None

def head(url: str, timeout: int = 15) -> Optional[Response]:
    """HEAD rápido para comprobar existencia/redirects sin descargar HTML."""
    _validate_url(url)
    headers = {"User-Agent": _pick_ua(), "Accept-Language": "es-ES,es;q=0.9,en;q=0.8"}
    try:
        return requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)
    except requests.RequestException:
        return None

def fetch_many_html(urls: Iterable[str],
                    concurrency: int = 4,
                    per_request_kwargs: Optional[Dict[str, Any]] = None) -> List[Tuple[str, str]]:
    """
    Descarga múltiples URLs en paralelo (ThreadPoolExecutor).
    Devuelve lista de tuplas (url, html).
    """
    per_request_kwargs = per_request_kwargs or {}
    results: List[Tuple[str, str]] = []
    urls_list = list(dict.fromkeys([u for u in urls if u]))

    def _worker(u: str) -> Tuple[str, str]:
        try:
            return u, fetch_html(u, **per_request_kwargs)
        except Exception:
            return u, ""

    if not urls_list:
        return results

    concurrency = max(1, min(concurrency, 10))
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(_worker, u): u for u in urls_list}
        for fut in as_completed(futures):
            u = futures[fut]
            try:
                url_ok, html = fut.result()
                results.append((url_ok, html))
            except Exception as e:
                logger.debug(f"Error en fetch_many_html({u}): {e}")
                results.append((u, ""))

    return results
