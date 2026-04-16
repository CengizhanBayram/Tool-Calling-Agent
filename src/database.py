"""Mock database layer — loads JSON files and exposes typed query methods.

All tool functions access data exclusively through :class:`MockDatabase`.
The module-level ``db`` singleton is imported by ``tools.py``.
"""

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional

from src.config import settings
from src.logger import log


class MockDatabase:
    """In-memory database backed by JSON files.

    Attributes:
        users: List of raw user dicts loaded from ``users.json``.
        transactions: List of raw transaction dicts.
        fraud_reasons: List of raw fraud-reason dicts.
    """

    def __init__(self, data_dir: Path) -> None:
        """Load all JSON data files from *data_dir*.

        Args:
            data_dir: Directory containing ``users.json``,
                      ``transactions.json``, and ``fraud_reasons.json``.
        """
        self._data_dir = data_dir
        self._lock = threading.RLock()
        self._load_all()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        """(Re-)load all three JSON files from disk."""
        self.users: List[Dict] = self._load("users.json")
        self.transactions: List[Dict] = self._load("transactions.json")
        self.fraud_reasons: List[Dict] = self._load("fraud_reasons.json")
        log.debug(
            f"DB loaded: {len(self.users)} users, "
            f"{len(self.transactions)} transactions, "
            f"{len(self.fraud_reasons)} fraud_reasons"
        )

    def _load(self, filename: str) -> List[Dict]:
        """Read a single JSON file and return its contents.

        Args:
            filename: Bare file name (no path prefix).

        Returns:
            Parsed list of dicts.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        path = self._data_dir / filename
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def find_user_by_email(self, email: str) -> Optional[Dict]:
        """Case-insensitive lookup of a user by email address.

        Args:
            email: The email to search for.

        Returns:
            The first matching user dict, or ``None`` if not found.
        """
        email_lower = email.strip().lower()
        with self._lock:
            return next(
                (u for u in self.users if u["email"].strip().lower() == email_lower),
                None,
            )

    def find_user_by_id(self, user_id: str) -> Optional[Dict]:
        """Look up a user by their internal user ID.

        Args:
            user_id: The user ID string (e.g. ``"USR-001"``).

        Returns:
            Matching user dict, or ``None``.
        """
        with self._lock:
            return next((u for u in self.users if u["user_id"] == user_id), None)

    def get_transactions_by_user(
        self,
        user_id: str,
        limit: int = 10,
        status_filter: Optional[str] = None,
    ) -> List[Dict]:
        """Retrieve transactions for a specific user, sorted newest-first.

        Args:
            user_id: Internal user ID.
            limit: Maximum number of records to return.
            status_filter: If provided, only return transactions with this
                           status (e.g. ``"failed"``).

        Returns:
            List of transaction dicts, capped at *limit*.
        """
        with self._lock:
            txns = [t for t in self.transactions if t["user_id"] == user_id]
        if status_filter:
            txns = [t for t in txns if t["status"] == status_filter]
        txns.sort(key=lambda t: t["created_at"], reverse=True)
        return txns[:limit]

    def get_fraud_reason(self, transaction_id: str) -> Optional[Dict]:
        """Retrieve the fraud record for a given transaction.

        Args:
            transaction_id: The transaction ID (e.g. ``"TXN-10041"``).

        Returns:
            Fraud-reason dict, or ``None`` if no record exists.
        """
        with self._lock:
            return next(
                (f for f in self.fraud_reasons if f["transaction_id"] == transaction_id),
                None,
            )

    def find_transaction_by_id(self, transaction_id: str) -> Optional[Dict]:
        """Find any transaction regardless of which user it belongs to.

        Args:
            transaction_id: The transaction ID.

        Returns:
            Transaction dict, or ``None``.
        """
        with self._lock:
            return next(
                (t for t in self.transactions if t["transaction_id"] == transaction_id),
                None,
            )

    def reload(self) -> None:
        """Re-read all JSON files from disk atomically.

        Safe to call at any time; in-flight reads hold the lock and will
        complete before the data is replaced.
        """
        with self._lock:
            self._load_all()
        log.info("Database reloaded from disk.")


# Module-level singleton imported by tools.py
db = MockDatabase(settings.data_dir)
