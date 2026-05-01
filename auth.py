"""Phone-number-based auth. Pure functions, no I/O."""

from __future__ import annotations
import hashlib
import os
import re

SALT = os.getenv("AUTH_SALT", "conduit-dev-salt-2024")


def normalize_phone(phone: str) -> str:
    """Strip everything except digits and leading +."""
    digits = re.sub(r"[^\d+]", "", phone or "")
    if digits and not digits.startswith("+"):
        digits = "+" + digits
    return digits


def phone_to_token(phone: str) -> str:
    normalized = normalize_phone(phone)
    return hashlib.sha256(f"{normalized}{SALT}".encode()).hexdigest()[:32]
