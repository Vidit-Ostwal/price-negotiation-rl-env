"""
generate_dataset.py
====================
Single entry point for dataset generation.

Usage:
    # Template mode (default) -- fast, no API calls
    python generators/generate_dataset.py --n 100 --output dataset.json

    # LLM mode -- unique products every episode, requires API
    python generators/generate_dataset.py --n 100 --mode llm --output dataset.json
"""

import os
import sys
import json
import uuid
import random
import argparse
from typing import Optional
from pathlib import Path

from generators.base import CATEGORIES
from generators.template import TemplateGenerator


def _load_dotenv(dotenv_path: str = ".env") -> None:
    """Load KEY=VALUE pairs into env when keys are missing or empty."""
    env_file = Path(dotenv_path)
    if not env_file.exists():
        return

    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and not os.getenv(key):
            os.environ[key] = value


def _validate_llm_env() -> dict:
    """
    Validate environment variables used by LLM generation mode.

    Required:
    - OPENAI_API_KEY

    Optional (defaults applied when unset):
    - OPENAI_API_BASE (default: https://api.openai.com/v1)
    - GENERATOR_MODEL (default: gpt-4o-mini)
    """
    _load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: missing required env var OPENAI_API_KEY for --mode llm")
        print("Set it in your shell or .env before running.")
        sys.exit(1)

    api_base = os.getenv("OPENAI_API_BASE")
    model = os.getenv("GENERATOR_MODEL")

    if not api_base:
        print("Info: OPENAI_API_BASE not set, defaulting to https://api.openai.com/v1")
        api_base = "https://api.openai.com/v1"

    if not model:
        print("Info: GENERATOR_MODEL not set, defaulting to gpt-4o-mini")
        model = "gpt-4o-mini"

    return {
        "api_key": api_key,
        "api_base": api_base,
        "model": model,
    }

# ---------------------------------------------------------------------------
# PROMPT TEMPLATES  (unchanged)
# ---------------------------------------------------------------------------

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
    "antiques":     "You know provenance, comparable auction results, and restoration costs. Serious collectors pay premium prices.",
    "electronics":  "You know current eBay sold listings, refurbishment costs, and that enthusiasts pay for authenticity.",
    "collectibles": "You understand grading standards, recent comparable sales, and how provenance affects price.",
    "vehicles":     "You know vehicle history, mechanic inspection results, and comparable listings.",
    "art":          "You understand attribution, period, technique, and comparable auction results.",
}

# ---------------------------------------------------------------------------
# VALUATION SAMPLER  (unchanged)
# ---------------------------------------------------------------------------

def sample_valuations(market_price: float) -> dict:
    buyer_value    = round(market_price * random.uniform(0.55, 0.90), -1)
    seller_reserve = round(market_price * random.uniform(0.45, 0.75), -1)

    deal_possible = buyer_value >= seller_reserve
    zopa_width    = max(0, buyer_value - seller_reserve)

    zopa_pct = zopa_width / market_price
    if zopa_pct > 0.3:   difficulty = "easy"
    elif zopa_pct > 0.1: difficulty = "medium"
    elif zopa_pct > 0:   difficulty = "hard"
    else:                difficulty = "no_deal"

    suggested_buyer_anchor = round(buyer_value * random.uniform(0.60, 0.75), -1)

    return {
        "buyer_true_value":       int(buyer_value),
        "seller_reserve_price":   int(seller_reserve),
        "market_price":           int(market_price),
        "zopa":                   [int(seller_reserve), int(buyer_value)] if deal_possible else None,
        "zopa_width":             int(zopa_width),
        "deal_possible":          deal_possible,
        "difficulty":             difficulty,
        "suggested_buyer_anchor": int(suggested_buyer_anchor),
    }

# ---------------------------------------------------------------------------
# EPISODE GENERATOR  (unchanged except product comes from generator)
# ---------------------------------------------------------------------------

def generate_episode(generator, category=None) -> dict:
    if category is None:
        category = random.choice(CATEGORIES)

    # The only line that changed -- generator replaces random.choice(PRODUCT_TEMPLATES)
    product           = generator.generate(category)
    product["category"] = category

    price_noise  = random.uniform(0.90, 1.10)
    market_price = round(product["market_price"] * price_noise, -1)
    valuations   = sample_valuations(market_price)

    buyer_prompt = BUYER_PROMPT_TEMPLATE.format(
        name=product["name"],
        description=product["description"],
        category=category,
        buyer_value=valuations["buyer_true_value"],
        market_price=valuations["market_price"],
    )

    seller_context = SELLER_CONTEXT_TEMPLATES.get(category, "")
    seller_prompt  = SELLER_PROMPT_TEMPLATE.format(
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
            "name":                 product["name"],
            "category":             category,
            "description":          product["description"],
            "market_price":         valuations["market_price"],
            "haggle_norm":          product["haggle_norm"],
            "typical_discount_pct": product["typical_discount_pct"],
        },
        "valuations":  valuations,
        "buyer_prompt":  buyer_prompt,
        "seller_prompt": seller_prompt,
        "information_asymmetry": {"seller_context": "full", "buyer_context": "sparse"},
        "metadata": {
            "max_turns":         10,
            "currency":          "USD",
            "generator_version": "2.0-llm" if hasattr(generator, "_call_llm") else "1.1-template",
        },
    }

# ---------------------------------------------------------------------------
# DATASET GENERATOR  (unchanged)
# ---------------------------------------------------------------------------

def generate_dataset(generator, n: int, balanced: bool = True) -> list:
    episodes = []

    if balanced:
        per_cat   = n // len(CATEGORIES)
        remainder = n % len(CATEGORIES)
        for i, cat in enumerate(CATEGORIES):
            count = per_cat + (1 if i < remainder else 0)
            for _ in range(count):
                episodes.append(generate_episode(generator, cat))
    else:
        for _ in range(n):
            episodes.append(generate_episode(generator))

    random.shuffle(episodes)

    difficulties = [e["valuations"]["difficulty"] for e in episodes]
    print(f"\n  Generated {len(episodes)} episodes")
    for d in ["easy", "medium", "hard", "no_deal"]:
        c = difficulties.count(d)
        print(f"  {d:10s}: {c:4d}  ({100*c/len(episodes):.1f}%)")

    return episodes

# ---------------------------------------------------------------------------
# MAIN  -- single entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate negotiation dataset")
    parser.add_argument("--n",        type=int,   default=100,            help="Number of episodes")
    parser.add_argument("--output",   type=str,   default="dataset.json", help="Output file path")
    parser.add_argument("--mode",     type=str,   default="template",     choices=["template", "llm"], help="Generator mode")
    parser.add_argument("--balanced", action="store_true", default=True,  help="Balance across categories")
    parser.add_argument("--seed",     type=int,   default=42,             help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Select generator
    if args.mode == "llm":
        from generators.llm import LLMGenerator
        llm_env = _validate_llm_env()
        generator = LLMGenerator(
            model=llm_env["model"],
            api_key=llm_env["api_key"],
            api_base=llm_env["api_base"],
        )
        print(f"Mode: LLM ({llm_env['model']}) -- each product uniquely generated")
    else:
        generator = TemplateGenerator()
        print("Mode: template -- using hardcoded product bank")

    print(f"Generating {args.n} episodes...")
    dataset = generate_dataset(generator, args.n, balanced=args.balanced)

    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n  Saved to {args.output}")

    ep = dataset[0]
    print(f"\n  Sample: {ep['product']['name']} ({ep['valuations']['difficulty']})")
    print(f"  Buyer ${ep['valuations']['buyer_true_value']} | Seller ${ep['valuations']['seller_reserve_price']} | ZOPA {ep['valuations']['zopa']}")
