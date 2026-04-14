"""
data/db_path.py — Single source of truth for the database path.

On local dev: uses articles.db (full database, gitignored)
On Streamlit Cloud: falls back to articles_deploy.db (committed, 0.4MB subset)
"""
import os
from pathlib import Path

_data_dir = Path(__file__).resolve().parent
_full     = _data_dir / "articles.db"
_deploy   = _data_dir / "articles_deploy.db"

DB_PATH = str(_full) if _full.exists() else str(_deploy)
