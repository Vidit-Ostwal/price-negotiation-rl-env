"""
generators/base.py — Abstract base + schema validation.
All generators must return a product dict matching REQUIRED_KEYS.
"""
from abc import ABC, abstractmethod

REQUIRED_KEYS = {"name", "description", "market_price", "haggle_norm", "typical_discount_pct"}
VALID_HAGGLE   = {"low", "medium", "high"}
CATEGORIES     = ["antiques", "electronics", "collectibles", "vehicles", "art"]


def validate_product(product: dict) -> dict:
    missing = REQUIRED_KEYS - set(product.keys())
    if missing:
        raise ValueError(f"Product missing keys: {missing}")
    if not isinstance(product["name"], str) or not product["name"].strip():
        raise ValueError("'name' must be a non-empty string")
    if not isinstance(product["description"], str) or not product["description"].strip():
        raise ValueError("'description' must be a non-empty string")
    if not (10 <= float(product["market_price"]) <= 100_000):
        raise ValueError(f"'market_price' out of range: {product['market_price']}")
    if product["haggle_norm"] not in VALID_HAGGLE:
        raise ValueError(f"'haggle_norm' must be one of {VALID_HAGGLE}")
    if not (0 <= int(product["typical_discount_pct"]) <= 50):
        raise ValueError(f"'typical_discount_pct' out of range: {product['typical_discount_pct']}")
    product["market_price"]         = int(product["market_price"])
    product["typical_discount_pct"] = int(product["typical_discount_pct"])
    return product


class ProductGenerator(ABC):
    @abstractmethod
    def generate(self, category: str) -> dict:
        """Return a validated product dict for the given category."""
        raise NotImplementedError
