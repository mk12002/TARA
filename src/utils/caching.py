"""
Caching Utilities
====================
Lightweight caching using functools.lru_cache + SQLite persistence.
No Redis needed.
"""

import json
import sqlite3
from functools import lru_cache
from hashlib import md5
from pathlib import Path
from typing import Optional

from loguru import logger


class SQLiteCache:
    """Simple key-value cache backed by SQLite."""

    def __init__(self, db_path: str = "./data/cache/tara_cache.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create cache table if it doesn't exist."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute("SELECT value FROM cache WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def set(self, key: str, value: str):
        """Set value in cache."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
            (key, value),
        )
        conn.commit()
        conn.close()

    def get_json(self, key: str) -> Optional[dict]:
        """Get JSON value from cache."""
        raw = self.get(key)
        return json.loads(raw) if raw else None

    def set_json(self, key: str, value: dict):
        """Set JSON value in cache."""
        self.set(key, json.dumps(value))

    def clear(self):
        """Clear all cache entries."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("DELETE FROM cache")
        conn.commit()
        conn.close()
        logger.info("Cache cleared")


def make_cache_key(*args) -> str:
    """Generate a deterministic cache key from arguments."""
    raw = "|".join(str(a) for a in args)
    return md5(raw.encode()).hexdigest()
