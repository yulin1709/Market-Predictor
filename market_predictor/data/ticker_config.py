"""
data/ticker_config.py — Single source of truth for commodity-to-ticker mappings.

All modules that need a yfinance ticker or Platts symbol must import from here.
Never hardcode ticker symbols elsewhere in the codebase.
"""

# ── Trading constants ─────────────────────────────────────────────────────────
MIN_CONFIDENCE_TO_TRADE = 0.50   # minimum model confidence to log a prediction
MIN_ARTICLES_TO_PREDICT = 5      # minimum articles needed before running inference
DEFAULT_LOT_SIZE = 1000          # barrels or MMBtu per lot for P&L calculation

# ── Commodity config ──────────────────────────────────────────────────────────
COMMODITY_CONFIG: dict[str, dict] = {
    # Full display names (used by streamlit_app.py commodity cards)
    "Dubai Crude (PCAAT00)": {
        "ticker":        "PCAAT00",
        "platts_symbol": "PCAAT00",
        "ticker_yf":     "CL=F",   # WTI futures — closest liquid proxy for Dubai
        "unit":          "barrel",
        "currency":      "USD",
    },
    "Brent Dated (PCAAS00)": {
        "ticker":        "PCAAS00",
        "platts_symbol": "PCAAS00",
        "ticker_yf":     "BZ=F",
        "unit":          "barrel",
        "currency":      "USD",
    },
    "WTI (PCACG00)": {
        "ticker":        "PCACG00",
        "platts_symbol": "PCACG00",
        "ticker_yf":     "CL=F",
        "unit":          "barrel",
        "currency":      "USD",
    },
    "LNG / Nat Gas (JKM)": {
        "ticker":        "JKM",
        "platts_symbol": "AAVSV00",  # Platts JKM symbol
        "ticker_yf":     "NG=F",
        "unit":          "MMBtu",
        "currency":      "USD",
    },
    # Short aliases for legacy rows and backfill
    "Dubai Crude": {
        "ticker":        "PCAAT00",
        "platts_symbol": "PCAAT00",
        "ticker_yf":     "CL=F",
        "unit":          "barrel",
        "currency":      "USD",
    },
    "Brent": {
        "ticker":        "PCAAS00",
        "platts_symbol": "PCAAS00",
        "ticker_yf":     "BZ=F",
        "unit":          "barrel",
        "currency":      "USD",
    },
    "WTI": {
        "ticker":        "PCACG00",
        "platts_symbol": "PCACG00",
        "ticker_yf":     "CL=F",
        "unit":          "barrel",
        "currency":      "USD",
    },
    "LNG": {
        "ticker":        "JKM",
        "platts_symbol": "AAVSV00",
        "ticker_yf":     "NG=F",
        "unit":          "MMBtu",
        "currency":      "USD",
    },
}

# ── Full display name aliases ─────────────────────────────────────────────────
DISPLAY_NAME_MAP: dict[str, str] = {
    "Dubai Crude (PCAAT00)": "Dubai Crude (PCAAT00)",
    "Brent Dated (PCAAS00)": "Brent Dated (PCAAS00)",
    "WTI (PCACG00)":         "WTI (PCACG00)",
    "LNG / Nat Gas (JKM)":   "LNG / Nat Gas (JKM)",
    "Dubai Crude":            "Dubai Crude (PCAAT00)",
    "Brent":                  "Brent Dated (PCAAS00)",
    "WTI":                    "WTI (PCACG00)",
    "LNG":                    "LNG / Nat Gas (JKM)",
}


def get_ticker(commodity: str) -> str | None:
    """Return the yfinance ticker symbol for a commodity, or None if unmapped."""
    cfg = COMMODITY_CONFIG.get(commodity)
    return cfg["ticker_yf"] if cfg else None


def get_platts_symbol(commodity: str) -> str | None:
    """Return the Platts symbol for a commodity, or None if unmapped."""
    cfg = COMMODITY_CONFIG.get(commodity)
    return cfg.get("platts_symbol") if cfg else None


def get_config(commodity: str) -> dict | None:
    """Return the full config dict for a commodity, or None if unmapped."""
    return COMMODITY_CONFIG.get(commodity)


def canonical_name(commodity: str) -> str:
    """Return the canonical full display name for a commodity."""
    return DISPLAY_NAME_MAP.get(commodity, commodity)


# ── Fundamental trader constants ──────────────────────────────────────────────

DIRECTION_LABELS: dict[str, dict] = {
    "bullish": {"icon": "▲", "label": "Bullish", "color": "#00C853"},
    "bearish": {"icon": "▼", "label": "Bearish", "color": "#F44336"},
    "neutral": {"icon": "→", "label": "Neutral",  "color": "#FFB300"},
    # legacy aliases
    "rise":    {"icon": "▲", "label": "Bullish", "color": "#00C853"},
    "fall":    {"icon": "▼", "label": "Bearish", "color": "#F44336"},
    "flat":    {"icon": "→", "label": "Neutral",  "color": "#FFB300"},
}

# Thesis invalidation level (%) — price moves this far against position → cut
STOP_LOSS_PCT: dict[str, float] = {
    "Dubai Crude (PCAAT00)": 1.5,
    "Brent Dated (PCAAS00)": 1.5,
    "WTI (PCACG00)":         1.5,
    "LNG / Nat Gas (JKM)":   2.0,
    "Dubai Crude":            1.5,
    "Brent":                  1.5,
    "WTI":                    1.5,
    "LNG":                    2.0,
}

IMPACT_SCORES: dict[str, int] = {
    "opec_decision":    10,
    "sanctions":         9,
    "force_majeure":     9,
    "pipeline_outage":   8,
    "supply_disruption": 8,
    "geopolitical":      7,
    "demand_shock":      7,
    "lng_outage":        7,
    "inventory_data":    6,
    "production_change": 6,
    "refinery_maintenance": 5,
    "shipping_disruption":  5,
    "demand_change":     4,
    "analyst_comment":   3,
    "routine_report":    2,
    "price_movement":    1,
    "general_market":    1,
    "other":             2,
}

MIN_IMPACT_SCORE_TO_TRADE = 6  # events scoring >= 6 qualify for trade signals

EVENT_TYPE_BADGES: dict[str, tuple[str, str]] = {
    "opec_decision":     ("OPEC Decision",     "#E8593C"),
    "sanctions":         ("Sanctions",          "#A32D2D"),
    "force_majeure":     ("Force Majeure",      "#A32D2D"),
    "pipeline_outage":   ("Pipeline Outage",    "#BA7517"),
    "supply_disruption": ("Supply Disruption",  "#BA7517"),
    "geopolitical":      ("Geopolitical",       "#534AB7"),
    "demand_shock":      ("Demand Shock",       "#185FA5"),
    "inventory_data":    ("Inventory Data",     "#0F6E56"),
    "lng_outage":        ("LNG Outage",         "#BA7517"),
    "refinery_maintenance": ("Refinery",        "#BA7517"),
    "shipping_disruption":  ("Tanker/Shipping", "#534AB7"),
}
