__version__ = "1.0.0"

# HTTP
from .http import (
    fetch_html,
    fetch_json,
    fetch_many_html,
    head,
)

# PDP parsing
from .parse_pdp import (
    assemble_product,
    parse_structured_product,
    parse_og,
    extract_heuristics,
    clean_text,
    parse_price_text,
)

# Shopping parsing/búsqueda
from .parse_shopping import (
    search_shopping,
    parse_shopping_html,
    parse_web_html,
    analyze_shopping_data,
)

# Matching
from .match import (
    MatchResult,
    match_one,
    match_greedy,
)

# Normalización
from .normalize import (
    parse_price,
    convert_currency,
    normalize_price_field,
    canonicalize_brand,
    canonicalize_model,
    normalize_specs,
)

# Insights/KPIs
from .insights import (
    build_insights_report,
    price_kpis_from_matches,
    market_profile,
    spec_coverage,
    variant_coverage,
)

__all__ = [
    # http
    "fetch_html", "fetch_json", "fetch_many_html", "head",
    # pdp
    "assemble_product", "parse_structured_product", "parse_og",
    "extract_heuristics", "clean_text", "parse_price_text",
    # shopping
    "search_shopping", "parse_shopping_html", "parse_web_html", "analyze_shopping_data",
    # matching
    "MatchResult", "match_one", "match_greedy",
    # normalize
    "parse_price", "convert_currency", "normalize_price_field",
    "canonicalize_brand", "canonicalize_model", "normalize_specs",
    # insights
    "build_insights_report", "price_kpis_from_matches", "market_profile",
    "spec_coverage", "variant_coverage",
]
