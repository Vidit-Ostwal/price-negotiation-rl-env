# Dataset Notes

This file documents the JSON schema produced by `generate_dataset.py` and consumed by `buyer_seller.py`.

## Episode Shape

Each element in `dataset.json` is one episode:

```json
{
  "episode_id": "...",
  "product": {...},
  "valuations": {...},
  "buyer_prompt": "...",
  "seller_prompt": "...",
  "information_asymmetry": {...},
  "metadata": {...}
}
```

## Top-Level Fields

- `episode_id`: UUID string
- `product`: item details shown to both parties
- `valuations`: hidden private values + ZOPA diagnostics
- `buyer_prompt`: buyer system prompt
- `seller_prompt`: seller system prompt
- `information_asymmetry`: context levels (`seller_context`, `buyer_context`)
- `metadata`: runtime metadata (`max_turns`, `currency`, `generator_version`)

## `product`

- `name`, `category`, `description`
- `market_price`: noisy market anchor used for this episode
- `haggle_norm`: category-level haggling intensity
- `typical_discount_pct`: category-level discount expectation

Categories in generator templates:

- `antiques`
- `electronics`
- `collectibles`
- `vehicles`
- `art`

## `valuations`

- `buyer_true_value`: buyer ceiling (sampled ~55–90% of market)
- `seller_reserve_price`: seller floor (sampled ~45–75% of market)
- `market_price`: same market anchor used in prompts
- `zopa`: `[seller_reserve_price, buyer_true_value]` when deal possible, else `null`
- `zopa_width`: `max(0, buyer_true_value - seller_reserve_price)`
- `deal_possible`: `buyer_true_value > seller_reserve_price`
- `difficulty`:
  - `easy` if `zopa_width / market_price > 0.3`
  - `medium` if `> 0.1`
  - `hard` if `> 0`
  - `no_deal` otherwise
- `suggested_buyer_anchor`: ~60–75% of buyer value (used by reward function, not shown to buyer)

## Prompts and Asymmetry

- Buyer prompt includes buyer private value and negotiation rules.
- Seller prompt includes seller reserve and richer category context.
- Current generator writes:
  - `information_asymmetry.seller_context = "full"`
  - `information_asymmetry.buyer_context = "sparse"`

## Metadata

- `max_turns`: currently `10` in generated episodes
- `currency`: `USD`
- `generator_version`: currently `1.1-template`

## How Environment Uses the Dataset

In `utils._load_hf_dataset()` each episode is transformed to:

- `prompt`: `[ {"role": "system", "content": buyer_prompt} ]`
- `info`: full original episode

In `buyer_seller._init_state()`, runtime state pulls from `info`:

- `buyer_true_value`, `seller_reserve`, `zopa_width`, `deal_possible`
- `buyer_prompt`, `seller_prompt`

## Reward-Relevant Dataset Fields

- `buyer_true_value`, `zopa_width`, `deal_possible` feed terminal rewards
- `metadata.max_turns` feeds efficiency bonus fallback logic
- `valuations.suggested_buyer_anchor` feeds anchoring reward

## Current Checked Sample (`dataset.json` in repo)

- 10 episodes
- balanced categories (2 each)
- difficulties include `easy`, `medium`, `hard`, and `no_deal`
- all episodes have `generator_version = "1.1-template"`
