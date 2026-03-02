import random
from generators.base import ProductGenerator, validate_product

PRODUCT_TEMPLATES = {
    "antiques": [
        {"name": "Victorian Mahogany Writing Desk", "description": "Late 1800s writing desk with brass hardware. Minor scratches.", "market_price": 850, "haggle_norm": "high", "typical_discount_pct": 25},
        {"name": "Art Deco Bronze Sculpture", "description": "1930s figurative bronze, signed by unknown artist. Stable patina.", "market_price": 1200, "haggle_norm": "high", "typical_discount_pct": 30},
        {"name": "Antique Persian Rug 6x9ft", "description": "Hand-knotted wool, 80 years old. Rich red and navy tones, minor edge wear.", "market_price": 2200, "haggle_norm": "high", "typical_discount_pct": 35},
        {"name": "Edwardian Silver Tea Set", "description": "5-piece sterling silver set, hallmarked 1908. Light tarnish.", "market_price": 1600, "haggle_norm": "high", "typical_discount_pct": 20},
    ],
    "electronics": [
        {"name": "Vintage Leica M3 Film Camera", "description": "1957 rangefinder, original leather case. Shutter fires, minor cosmetic wear.", "market_price": 1100, "haggle_norm": "medium", "typical_discount_pct": 15},
        {"name": "Apple PowerBook G4 Titanium", "description": "2001 laptop, rare collector piece. Powers on, original box included.", "market_price": 400, "haggle_norm": "medium", "typical_discount_pct": 20},
        {"name": "Vintage Sennheiser HD 414 Headphones", "description": "1970s open-back headphones. Foam pads replaced, cable intact.", "market_price": 220, "haggle_norm": "medium", "typical_discount_pct": 15},
        {"name": "Pioneer SX-1250 Stereo Receiver", "description": "1976 flagship receiver, 160W per channel. Recently serviced.", "market_price": 1800, "haggle_norm": "medium", "typical_discount_pct": 20},
    ],
    "collectibles": [
        {"name": "1952 Topps Mickey Mantle Card", "description": "Graded PSA 3. Iconic rookie-era card. Light crease on corner.", "market_price": 5000, "haggle_norm": "high", "typical_discount_pct": 10},
        {"name": "First Edition The Sun Also Rises", "description": "1926 first printing, dust jacket torn. Boards tight, foxing on endpapers.", "market_price": 3200, "haggle_norm": "high", "typical_discount_pct": 15},
        {"name": "NASA Mission Patch Collection 1960s", "description": "12 original cloth patches, Mercury through Apollo. Minor soiling on two.", "market_price": 750, "haggle_norm": "medium", "typical_discount_pct": 20},
        {"name": "Signed Pele Football Shirt 1970", "description": "Brazil jersey signed in black marker. COA included. Minor fading.", "market_price": 4500, "haggle_norm": "high", "typical_discount_pct": 10},
    ],
    "vehicles": [
        {"name": "1987 Porsche 944", "description": "Original paint, 112k miles. Recent timing belt. Some undercarriage rust.", "market_price": 12000, "haggle_norm": "high", "typical_discount_pct": 15},
        {"name": "1965 Vespa GS 160", "description": "Fully restored 4 years ago. Repainted in original grey. Runs well.", "market_price": 6500, "haggle_norm": "high", "typical_discount_pct": 20},
    ],
    "art": [
        {"name": "Oil Painting French Countryside 1940s", "description": "Unsigned impressionist landscape, 24x36 inches. Original gilt frame. Minor craquelure.", "market_price": 1800, "haggle_norm": "high", "typical_discount_pct": 30},
        {"name": "Bernard Buffet Lithograph", "description": "Signed and numbered 42/150. Clown series 1967. Some yellowing on margins.", "market_price": 2800, "haggle_norm": "high", "typical_discount_pct": 25},
    ],
}


class TemplateGenerator(ProductGenerator):
    def generate(self, category: str) -> dict:
        if category not in PRODUCT_TEMPLATES:
            raise ValueError(f"Unknown category: {category}")
        product = random.choice(PRODUCT_TEMPLATES[category]).copy()
        validate_product(product)
        return product
