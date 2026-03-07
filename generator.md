# Generator Guide

This repository uses `generators/generate_dataset.py` as the dataset generation entrypoint.

## Modes

- `template` (default): picks products from `generators/template.py`
- `llm`: generates products via `generators/llm.py`; on repeated LLM failure it falls back to template

## Run

Template mode:

```bash
uv run python generators/generate_dataset.py --n 100 --mode template --output dataset.json --seed 42
```

LLM mode:

```bash
OPENAI_API_KEY=sk-... \
OPENAI_API_BASE=https://api.openai.com/v1 \
GENERATOR_MODEL=gpt-4o-mini \
uv run python generators/generate_dataset.py --n 100 --mode llm --output dataset.json --seed 42
```

Generate and push directly to Hugging Face Hub:

```bash
HF_TOKEN=hf_... \
HF_DATASET_REPO=your-hf-username/price-negotiation-dataset \
uv run python generators/generate_dataset.py \
  --mode llm --n 100 --output dataset.json --seed 42 \
  --push-to-hf --hf-split train
```

Notes:
- `OPENAI_API_KEY` is required for `--mode llm`
- `OPENAI_API_BASE` defaults to `https://api.openai.com/v1`
- `GENERATOR_MODEL` defaults to `gpt-4o-mini`
- missing values are auto-loaded from repo-root `.env` before validation
- balancing is effectively on by default in current CLI behavior
- use `--unbalanced` to disable category balancing
- for HF push, token can be provided by `--hf-token` or env (`HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`)
- for HF push, repo can be provided by `--hf-repo-id` or env (`HF_DATASET_REPO` / `HF_REPO_ID`)

## Hugging Face Push (CLI)

`generators/generate_dataset.py` now supports pushing the generated dataset split directly to Hugging Face Hub.

Flags:
- `--push-to-hf`: enable push after local JSON write
- `--hf-repo-id`: dataset repo id (`username/repo`)
- `--hf-token`: HF token (otherwise from env)
- `--hf-split`: split name (default `train`)
- `--hf-private`: create/update private repo
- `--hf-write-mode`: `append` (default) or `overwrite`
- `--hf-push-every`: for `--mode llm` + `--push-to-hf`, append every N rows (default `100`)
- `--hf-commit-message`: custom commit message

Examples:
- append to existing split:
  - `--push-to-hf --hf-split train` (default behavior)
- checkpoint append every 100 rows (default for LLM push):
  - `--mode llm --push-to-hf --hf-split train`
- checkpoint append every 50 rows:
  - `--mode llm --push-to-hf --hf-split train --hf-push-every 50`
- overwrite target split:
  - `--push-to-hf --hf-split train --hf-write-mode overwrite`

## Generation Pipeline (Code-Level)

1. `main` in `generators/generate_dataset.py` selects generator:
- `TemplateGenerator()` for template mode
- `LLMGenerator(...)` for llm mode

2. `generate_dataset(generator, n, balanced)` builds episode list:
- balanced: split count across `CATEGORIES`
- unbalanced: random category each episode

3. `generate_episode(generator, category)` builds one episode:
- gets base product from `generator.generate(category)`
- applies market noise
- calls `sample_valuations(market_price)`
- formats buyer/seller prompts
- assembles final episode JSON

4. Product validation is enforced by `validate_product` in `generators/base.py`:
- required keys present
- field types/ranges validated

## Field-by-Field Attribute Reference

## Top-Level

- `episode_id`
  - Meaning: unique identifier for one negotiation instance.
  - Source: `str(uuid.uuid4())` in `generate_episode`.

- `product`
  - Meaning: market item attributes shared in prompts.
  - Source: generator output + normalized fields.

- `valuations`
  - Meaning: hidden economics for buyer/seller and reward signals.
  - Source: `sample_valuations(...)`.

- `buyer_prompt`, `seller_prompt`
  - Meaning: system instructions used during rollout.
  - Source: `BUYER_PROMPT_TEMPLATE.format(...)`, `SELLER_PROMPT_TEMPLATE.format(...)`.

- `information_asymmetry`
  - Meaning: who has richer context.
  - Source: hardcoded in `generate_episode`.

- `metadata`
  - Meaning: runtime constants/versioning.
  - Source: hardcoded in `generate_episode`.

## `product` fields

- `name`
  - Meaning: concrete listing title.
  - Source:
    - template mode: from `PRODUCT_TEMPLATES[category]`
    - llm mode: from LLM JSON output

- `category`
  - Meaning: one of `antiques|electronics|collectibles|vehicles|art`.
  - Source: input category passed to `generate_episode`.

- `description`
  - Meaning: negotiable condition details shown to both sides.
  - Source: template text or LLM-generated text.

- `market_price`
  - Meaning: per-episode noisy market anchor used everywhere in prompts/valuation.
  - Formula:
    - base product price: `product["market_price"]`
    - noise: `price_noise ~ Uniform(0.90, 1.10)`
    - episode market: `round(base * price_noise, -1)`

- `haggle_norm`
  - Meaning: expected haggling intensity.
  - Source: template/LLM product field.
  - Valid values: `low|medium|high`.

- `typical_discount_pct`
  - Meaning: category-level expected discount pressure in seller prompt.
  - Source: template/LLM product field.
  - Validation range: integer `0..50` (base validator).

## `valuations` fields (mathematical definitions)

All are computed in `sample_valuations(market_price)`.

Let `M = market_price`.

- `buyer_true_value`
  - Meaning: buyer's private max willingness-to-pay.
  - Formula: `B = round(M * U(0.55, 0.90), -1)`.

- `seller_reserve_price`
  - Meaning: seller's private minimum acceptable price.
  - Formula: `S = round(M * U(0.45, 0.75), -1)`.

- `market_price`
  - Meaning: repeats episode market anchor for convenience.
  - Formula: `M` (integer).

- `deal_possible`
  - Meaning: whether generator labels overlap as deal-capable.
  - Formula in current code: `B >= S`.

- `zopa`
  - Meaning: zone of possible agreement interval.
  - Formula: `[S, B] if deal_possible else null`.

- `zopa_width`
  - Meaning: width of feasible overlap.
  - Formula: `max(0, B - S)`.

- `difficulty`
  - Meaning: difficulty bucket by normalized overlap.
  - Define `r = zopa_width / M`.
  - Rules:
    - `easy` if `r > 0.3`
    - `medium` if `r > 0.1`
    - `hard` if `r > 0`
    - `no_deal` otherwise

- `suggested_buyer_anchor`
  - Meaning: suggested opening level used by reward shaping.
  - Formula: `round(B * U(0.60, 0.75), -1)`.

## Prompt fields

- `buyer_prompt`
  - Built from item details + `buyer_true_value` + `market_price`.
  - Includes required action grammar:
    - `<action>OFFER $X</action>`
    - `<action>ACCEPT</action>`
    - `<action>WALK</action>`

- `seller_prompt`
  - Built from item details + `seller_reserve_price` + `market_price` + `typical_discount_pct` + category context.

## `information_asymmetry`

- `seller_context = "full"`
- `buyer_context = "sparse"`

This is currently static in generated output.

## `metadata`

- `max_turns = 10`
- `currency = "USD"`
- `generator_version`
  - `"2.0-llm"` when generator has `_call_llm` (LLM mode)
  - otherwise `"1.1-template"`

## LLM Mode Details

`LLMGenerator._call_llm(...)` asks model for strict JSON with:
- `name`
- `description`
- `market_price`
- `haggle_norm`
- `typical_discount_pct`

Then:
- strips code fences if present
- parses JSON
- validates via `validate_product`
- retries up to `max_retries`
- falls back to `TemplateGenerator` if still failing

## Output Shape Example

```json
{
  "episode_id": "uuid",
  "product": {
    "name": "item",
    "category": "electronics",
    "description": "...",
    "market_price": 1230,
    "haggle_norm": "medium",
    "typical_discount_pct": 20
  },
  "valuations": {
    "buyer_true_value": 980,
    "seller_reserve_price": 760,
    "market_price": 1230,
    "zopa": [760, 980],
    "zopa_width": 220,
    "deal_possible": true,
    "difficulty": "medium",
    "suggested_buyer_anchor": 650
  },
  "buyer_prompt": "...",
  "seller_prompt": "...",
  "information_asymmetry": {
    "seller_context": "full",
    "buyer_context": "sparse"
  },
  "metadata": {
    "max_turns": 10,
    "currency": "USD",
    "generator_version": "1.1-template"
  }
}
```

## Compatibility with Environment

Generated episodes are compatible with current runtime:
- `utils._load_hf_dataset()` reads each episode and wraps it into `prompt` + `info`
- `buyer_seller._init_state()` consumes `valuations`, `buyer_prompt`, `seller_prompt`
- rewards in `rewards.py` use `buyer_true_value`, `zopa_width`, `deal_possible`, `metadata.max_turns`, `suggested_buyer_anchor`

Semantic caveat:
- generator uses `deal_possible = (B >= S)`
- some other logic may expect strict `B > S`
- if you want strict semantics, change that expression in `sample_valuations`
