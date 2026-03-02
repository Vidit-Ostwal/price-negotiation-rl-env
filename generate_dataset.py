"""
Synthetic Negotiation Dataset Generator
========================================
Generates N negotiation episodes using templates.
Each episode has a product, buyer prompt, seller prompt,
valuations, ZOPA, and metadata.

Usage:
    python generate_dataset.py --n 100 --output dataset.json
"""

import json
import uuid
import random          # fixed: was "randoma"
import argparse
from typing import Optional

# ---------------------------------------------------------------------------
# 1. PRODUCT TEMPLATES
# ---------------------------------------------------------------------------

PRODUCT_TEMPLATES = {
    "antiques": [
        {
            "name": "Victorian Mahogany Writing Desk",
            "description": "Late 1800s writing desk with brass hardware and original felt lining. Minor scratches on surface.",
            "market_price": 850,
            "haggle_norm": "high",
            "typical_discount_pct": 25,
        },
        {
            "name": "Art Deco Bronze Sculpture",
            "description": "1930s figurative bronze, signed by unknown artist. 14 inches tall, stable patina.",
            "market_price": 1200,
            "haggle_norm": "high",
            "typical_discount_pct": 30,
        },
        {
            "name": "Antique Persian Rug (6x9 ft)",
            "description": "Hand-knotted wool rug, estimated 80 years old. Rich red and navy tones, minor wear on edges.",
            "market_price": 2200,
            "haggle_norm": "high",
            "typical_discount_pct": 35,
        },
        {
            "name": "Edwardian Silver Tea Set",
            "description": "5-piece sterling silver set, hallmarked 1908. Teapot, sugar, creamer, tray, tongs. Light tarnish.",
            "market_price": 1600,
            "haggle_norm": "high",
            "typical_discount_pct": 20,
        },
    ],
    "electronics": [
        {
            "name": "Vintage Leica M3 Film Camera",
            "description": "1957 rangefinder camera, original leather case. Shutter fires correctly, minor cosmetic wear.",
            "market_price": 1100,
            "haggle_norm": "medium",
            "typical_discount_pct": 15,
        },
        {
            "name": "Mint Apple PowerBook G4",
            "description": "Titanium 2001 laptop, rare collector piece. Powers on, original box included.",
            "market_price": 400,
            "haggle_norm": "medium",
            "typical_discount_pct": 20,
        },
        {
            "name": "Vintage Sennheiser HD 414 Headphones",
            "description": "1970s open-back headphones. Foam pads replaced, cable intact. Classic hi-fi sound.",
            "market_price": 220,
            "haggle_norm": "medium",
            "typical_discount_pct": 15,
        },
        {
            "name": "Pioneer SX-1250 Stereo Receiver",
            "description": "1976 flagship receiver, 160W per channel. Recently serviced, all functions working.",
            "market_price": 1800,
            "haggle_norm": "medium",
            "typical_discount_pct": 20,
        },
    ],
    "collectibles": [
        {
            "name": "1952 Topps Mickey Mantle Baseball Card",
            "description": "Graded PSA 3 (VG). Iconic rookie-era card. Light crease on corner, centered well.",
            "market_price": 5000,
            "haggle_norm": "high",
            "typical_discount_pct": 10,
        },
        {
            "name": "First Edition Hemingway 'The Sun Also Rises'",
            "description": "1926 first printing, dust jacket present but torn. Boards tight, foxing on endpapers.",
            "market_price": 3200,
            "haggle_norm": "high",
            "typical_discount_pct": 15,
        },
        {
            "name": "Vintage 1960s NASA Mission Patch Collection",
            "description": "Set of 12 original cloth patches, Mercury through Apollo. Minor soiling on two patches.",
            "market_price": 750,
            "haggle_norm": "medium",
            "typical_discount_pct": 20,
        },
        {
            "name": "Signed Pelé Football Shirt (1970 World Cup)",
            "description": "Brazil yellow jersey signed in black marker. COA included. Framed, minor fading.",
            "market_price": 4500,
            "haggle_norm": "high",
            "typical_discount_pct": 10,
        },
    ],
    "vehicles": [
        {
            "name": "1987 Porsche 944",
            "description": "Original paint, 112k miles. Recent timing belt. Some rust on undercarriage, drives well.",
            "market_price": 12000,
            "haggle_norm": "high",
            "typical_discount_pct": 15,
        },
        {
            "name": "Vintage 1965 Vespa GS 160",
            "description": "Italian scooter, fully restored 4 years ago. Repainted in original grey. Runs well.",
            "market_price": 6500,
            "haggle_norm": "high",
            "typical_discount_pct": 20,
        },
    ],
    "art": [
        {
            "name": "Oil Painting — French Countryside (1940s)",
            "description": "Unsigned impressionist landscape, 24x36 inches. Oil on canvas, original gilt frame. Minor craquelure.",
            "market_price": 1800,
            "haggle_norm": "high",
            "typical_discount_pct": 30,
        },
        {
            "name": "Lithograph by Bernard Buffet",
            "description": "Signed and numbered 42/150. Clown series, 1967. Some yellowing on margins, unframed.",
            "market_price": 2800,
            "haggle_norm": "high",
            "typical_discount_pct": 25,
        },
    ],
}

# ---------------------------------------------------------------------------
# 2. PROMPT TEMPLATES
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
Negotiate the lowest possible price. Be strategic — use anchoring, express hesitation,
highlight flaws, and make incremental concessions. Do not accept the first offer.

RULES:
- You MUST respond with one of these actions:
  <action>OFFER $X</action>  — make or counter with a specific dollar offer
  <action>ACCEPT</action>    — accept the current offer on the table
  <action>WALK</action>      — walk away from the negotiation
- You may include reasoning before your action tag.
- Never reveal your true maximum value of ${buyer_value}."""

SELLER_PROMPT_TEMPLATE = """You are an expert dealer negotiating to SELL the following item.

ITEM: {name}
DESCRIPTION: {description}
CATEGORY: {category}

YOUR PRIVATE INFORMATION (never reveal this):
- Your absolute minimum (reserve price) is ${seller_reserve}.
- You know the fair market value is around ${market_price}.
- You have sold many items like this. Typical buyers discount by {typical_discount_pct}%.

YOUR EXPERTISE:
{seller_context}

YOUR GOAL:
Sell for as high a price as possible, ideally close to market value.
Use anchoring high, emphasize provenance and quality, create mild urgency,
and make small strategic concessions to appear reasonable.

RULES:
- You MUST respond with one of these actions:
  <action>OFFER $X</action>  — state your asking or counter price
  <action>ACCEPT</action>    — accept the current offer on the table
  <action>WALK</action>      — walk away if offer is below your reserve
- You may include reasoning / sales pitch before your action tag.
- Never reveal your reserve price of ${seller_reserve}."""

SELLER_CONTEXT_TEMPLATES = {
    "antiques": "You know this piece's provenance, comparable auction results, and restoration costs. You're aware that serious collectors pay premium prices and that scarcity drives value in this category.",
    "electronics": "You know the current eBay sold listings, what refurbishment cost, and that vintage electronics buyers are often enthusiasts willing to pay for authenticity and working condition.",
    "collectibles": "You understand grading standards, recent comparable sales, and that provenance documentation significantly affects price. You know how to highlight scarcity and investment value.",
    "vehicles": "You know the vehicle history, recent mechanic inspection results, and comparable listings. You understand that emotional attachment and rarity can push prices above book value.",
    "art": "You understand attribution, period, technique, and comparable auction results. You know how to frame artistic merit and investment potential for buyers.",
}

# ---------------------------------------------------------------------------
# 3. VALUATION SAMPLER
# ---------------------------------------------------------------------------

def sample_valuations(market_price: float, category: str) -> dict:
    """
    Sample buyer and seller valuations around the market price.

    Buyer true value: 55–90% of market price.
        A rational buyer on a second-hand market always wants below market.
        Sampling above market (old range 70–110%) was wrong — it inflated
        the ZOPA artificially and made reward signals misleading.

    Seller reserve: 45–75% of market price.
        Seller's cost basis / floor. Always below buyer's ceiling so a
        ZOPA can realistically exist.
    """

    # fixed: was 0.70–1.10 which allowed buyer to pay above market
    buyer_value    = round(market_price * random.uniform(0.55, 0.90), -1)

    # fixed: tightened upper bound from 0.85 → 0.75 to keep seller
    # reserve realistically below buyer value
    seller_reserve = round(market_price * random.uniform(0.45, 0.75), -1)

    deal_possible = buyer_value > seller_reserve
    zopa_width    = max(0, buyer_value - seller_reserve)

    zopa_pct = zopa_width / market_price
    if zopa_pct > 0.3:
        difficulty = "easy"
    elif zopa_pct > 0.1:
        difficulty = "medium"
    elif zopa_pct > 0:
        difficulty = "hard"
    else:
        difficulty = "no_deal"

    anchor_pct             = random.uniform(0.60, 0.75)
    suggested_buyer_anchor = round(buyer_value * anchor_pct, -1)

    return {
        "buyer_true_value":      int(buyer_value),
        "seller_reserve_price":  int(seller_reserve),
        "market_price":          int(market_price),
        "zopa":                  [int(seller_reserve), int(buyer_value)] if deal_possible else None,
        "zopa_width":            int(zopa_width),
        "deal_possible":         deal_possible,
        "difficulty":            difficulty,
        "suggested_buyer_anchor": int(suggested_buyer_anchor),
    }

# ---------------------------------------------------------------------------
# 4. INFORMATION ASYMMETRY LEVELS
# ---------------------------------------------------------------------------

def get_buyer_context_level(difficulty: str) -> str:
    """
    Controls how much product knowledge the buyer prompt includes.
    Sparse = weaker buyer, which is our default training scenario.
    """
    return "sparse"

# ---------------------------------------------------------------------------
# 5. EPISODE GENERATOR
# ---------------------------------------------------------------------------

def generate_episode(category: Optional[str] = None) -> dict:
    """Generate a single negotiation episode."""

    if category is None:
        category = random.choice(list(PRODUCT_TEMPLATES.keys()))

    product = random.choice(PRODUCT_TEMPLATES[category]).copy()
    product["category"] = category

    price_noise  = random.uniform(0.90, 1.10)
    market_price = round(product["market_price"] * price_noise, -1)

    valuations = sample_valuations(market_price, category)

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
        "valuations": valuations,
        "buyer_prompt":  buyer_prompt,
        "seller_prompt": seller_prompt,
        "information_asymmetry": {
            "seller_context": "full",
            "buyer_context":  "sparse",
        },
        "metadata": {
            "max_turns":         10,
            "currency":          "USD",
            "generator_version": "1.1-template",  # bumped for valuation fix
        },
    }

# ---------------------------------------------------------------------------
# 6. DATASET GENERATOR
# ---------------------------------------------------------------------------

def generate_dataset(n: int, balanced: bool = True) -> list:
    episodes   = []
    categories = list(PRODUCT_TEMPLATES.keys())

    if balanced:
        per_category = n // len(categories)
        remainder    = n % len(categories)
        for i, category in enumerate(categories):
            count = per_category + (1 if i < remainder else 0)
            for _ in range(count):
                episodes.append(generate_episode(category))
    else:
        for _ in range(n):
            episodes.append(generate_episode())

    random.shuffle(episodes)

    difficulties = [e["valuations"]["difficulty"] for e in episodes]
    cats         = [e["product"]["category"]      for e in episodes]

    print(f"\n✅ Generated {len(episodes)} episodes")
    print(f"\nDifficulty distribution:")
    for d in ["easy", "medium", "hard", "no_deal"]:
        count = difficulties.count(d)
        print(f"  {d:10s}: {count:4d}  ({100*count/len(episodes):.1f}%)")

    print(f"\nCategory distribution:")
    for c in categories:
        count = cats.count(c)
        print(f"  {c:15s}: {count:4d}  ({100*count/len(episodes):.1f}%)")

    return episodes

# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic negotiation dataset")
    parser.add_argument("--n",        type=int, default=100,          help="Number of episodes to generate")
    parser.add_argument("--output",   type=str, default="dataset.json", help="Output file path")
    parser.add_argument("--balanced", action="store_true", default=True, help="Balance across categories")
    parser.add_argument("--seed",     type=int, default=42,           help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Generating {args.n} negotiation episodes...")
    dataset = generate_dataset(args.n, balanced=args.balanced)

    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n💾 Saved to {args.output}")

    print("\n--- SAMPLE EPISODE PREVIEW ---")
    ep = dataset[0]
    print(f"Product   : {ep['product']['name']}")
    print(f"Category  : {ep['product']['category']}")
    print(f"Market $  : ${ep['valuations']['market_price']}")
    print(f"Buyer val : ${ep['valuations']['buyer_true_value']}")
    print(f"Seller res: ${ep['valuations']['seller_reserve_price']}")
    print(f"ZOPA      : {ep['valuations']['zopa']}")
    print(f"Difficulty: {ep['valuations']['difficulty']}")
    print(f"Anchor    : ${ep['valuations']['suggested_buyer_anchor']}")
    print(f"\nBUYER PROMPT:\n{ep['buyer_prompt'][:300]}...")
    print(f"\nSELLER PROMPT:\n{ep['seller_prompt'][:300]}...")