"""
Episode construction utilities for the buyer-seller negotiation dataset.

This module contains:
- Prompt templates for buyer and seller
- Category-specific seller context snippets
- Valuation sampling logic
- The core `generate_episode(...)` helper used by dataset generation
"""

import random
import uuid

from generators.base import CATEGORIES


BUYER_PROMPT_TEMPLATE = """You are negotiating to BUY the following item from a private seller.

ITEM: {name}
DESCRIPTION: {description}
CATEGORY: {category}

YOUR PRIVATE INFORMATION (never reveal this):
- The maximum you are willing to pay is ${buyer_value}.
- You believe the fair market value is around ${market_price}.
- If you pay less than ${buyer_value}, you make a profit. The lower the price, the better.

YOUR GOAL:
Negotiate the lowest possible price. Be strategic -- use anchoring, express hesitation,
highlight flaws, and make incremental concessions. Do not accept the first offer.

RULES:
- You MUST respond with one of these actions:
  <action>OFFER $X</action>  -- make or counter with a specific dollar offer
  <action>ACCEPT</action>    -- accept the current offer on the table
  <action>WALK</action>      -- walk away from the negotiation
- You may include reasoning before your action tag.
- Never reveal your true maximum value of ${buyer_value}."""


SELLER_PROMPT_TEMPLATE = """You are an expert dealer negotiating to SELL the following item.

ITEM: {name}
DESCRIPTION: {description}
CATEGORY: {category}

YOUR PRIVATE INFORMATION (never reveal this):
- Your absolute minimum (reserve price) is ${seller_reserve}.
- You know the fair market value is around ${market_price}.
- Typical buyers discount by {typical_discount_pct}%.

YOUR EXPERTISE:
{seller_context}

YOUR GOAL:
Sell for as high a price as possible, ideally close to market value.
Use anchoring high, emphasize provenance and quality, create mild urgency,
and make small strategic concessions to appear reasonable.

RULES:
- You MUST respond with one of these actions:
  <action>OFFER $X</action>  -- state your asking or counter price
  <action>ACCEPT</action>    -- accept the current offer on the table
  <action>WALK</action>      -- walk away if offer is below your reserve
- You may include reasoning before your action tag.
- Never reveal your reserve price of ${seller_reserve}."""


SELLER_CONTEXT_TEMPLATES = {
    "antiques": "You know provenance, comparable auction results, and restoration costs. Serious collectors pay premium prices.",
    "electronics": "You know current eBay sold listings, refurbishment costs, and that enthusiasts pay for authenticity.",
    "collectibles": "You understand grading standards, recent comparable sales, and how provenance affects price.",
    "vehicles": "You know vehicle history, mechanic inspection results, and comparable listings.",
    "art": "You understand attribution, period, technique, and comparable auction results.",
    "furniture": "You know brand premiums, construction quality, delivery friction, and recent local marketplace comps.",
    "jewelry": "You understand metal purity, gemstone quality, hallmark verification, and retail versus resale pricing.",
    "musical_instruments": "You know model-year desirability, maintenance history, playability issues, and used-market comps.",
    "sports_outdoors": "You know seasonal demand, equipment wear patterns, upgrade value, and local resale pricing.",
    "luxury_fashion": "You understand authentication details, condition grading, release popularity, and second-hand platform comps.",
}


def sample_valuations(market_price: float) -> dict:
    """Sample buyer/seller private values and derived difficulty labels."""
    buyer_value = round(market_price * random.uniform(0.55, 0.90), -1)
    seller_reserve = round(market_price * random.uniform(0.45, 0.75), -1)

    deal_possible = buyer_value >= seller_reserve
    zopa_width = max(0, buyer_value - seller_reserve)

    zopa_pct = zopa_width / market_price
    if zopa_pct > 0.3:
        difficulty = "easy"
    elif zopa_pct > 0.1:
        difficulty = "medium"
    elif zopa_pct > 0:
        difficulty = "hard"
    else:
        difficulty = "no_deal"

    suggested_buyer_anchor = round(buyer_value * random.uniform(0.60, 0.75), -1)

    return {
        "buyer_true_value": int(buyer_value),
        "seller_reserve_price": int(seller_reserve),
        "market_price": int(market_price),
        "zopa": [int(seller_reserve), int(buyer_value)] if deal_possible else None,
        "zopa_width": int(zopa_width),
        "deal_possible": deal_possible,
        "difficulty": difficulty,
        "suggested_buyer_anchor": int(suggested_buyer_anchor),
    }


def generate_episode(generator, category: str | None = None) -> dict:
    """Build a single episode dict using the provided product generator."""
    if category is None:
        category = random.choice(CATEGORIES)

    product = generator.generate(category)
    product["category"] = category

    price_noise = random.uniform(0.90, 1.10)
    market_price = round(product["market_price"] * price_noise, -1)
    valuations = sample_valuations(market_price)

    buyer_prompt = BUYER_PROMPT_TEMPLATE.format(
        name=product["name"],
        description=product["description"],
        category=category,
        buyer_value=valuations["buyer_true_value"],
        market_price=valuations["market_price"],
    )

    seller_context = SELLER_CONTEXT_TEMPLATES.get(category, "")
    seller_prompt = SELLER_PROMPT_TEMPLATE.format(
        name=product["name"],
        description=product["description"],
        category=category,
        seller_reserve=valuations["seller_reserve_price"],
        market_price=valuations["market_price"],
        typical_discount_pct=product["typical_discount_pct"],
        seller_context=seller_context,
    )

    return {
        "episode_id": str(uuid.uuid4()),
        "product": {
            "name": product["name"],
            "category": category,
            "description": product["description"],
            "market_price": valuations["market_price"],
            "haggle_norm": product["haggle_norm"],
            "typical_discount_pct": product["typical_discount_pct"],
        },
        "valuations": valuations,
        "buyer_prompt": buyer_prompt,
        "seller_prompt": seller_prompt,
        "information_asymmetry": {"seller_context": "full", "buyer_context": "sparse"},
        "metadata": {
            "max_turns": 10,
            "currency": "USD",
            "generator_version": (
                "2.1-hf-llm"
                if hasattr(generator, "_call_hf")
                else "2.0-llm"
                if hasattr(generator, "_call_llm")
                else "1.1-template"
            ),
        },
    }
