# buyer-seller

Multi-turn buyer/seller negotiation environment for `verifiers`.

- Buyer: policy model selected by `vf-eval -m ...`
- Seller: fixed opponent called inside the environment via LiteLLM

## Repository Layout

- `buyer_seller.py`: `NegotiationEnv` + `load_environment()` entrypoint
- `utils.py`: `.env` loading, env validation, dataset loading, role-flip helper
- `rewards.py`: action parser + 7 reward functions
- `generate_dataset.py`: synthetic dataset generator
- `dataset.json`: sample dataset (10 episodes)
- `test_seller_model_smoke.py`: real seller API smoke test (no mocks)
- `understanding_dataset.md`: field-by-field dataset schema notes

## Configuration

The environment uses `utils._validate_env()` and auto-loads `.env` from repo root.

Required:

- `OPENAI_API_KEY`
- `SELLER_MODEL`
- `OPENAI_API_BASE`
- `DATASET_PATH`

Optional:

- `MAX_TURNS` (default `10`)

Example `.env`:

```bash
OPENAI_API_KEY=sk-...
SELLER_MODEL=openai/gpt-4.1-mini
OPENAI_API_BASE=https://api.openai.com/v1
DATASET_PATH=dataset.json
MAX_TURNS=10
```

## Runtime Flow

1. `load_environment()` validates env vars and loads dataset.
2. Buyer sends an action (`<action>OFFER $X</action>`, `ACCEPT`, or `WALK`).
3. Env parses buyer action and updates state.
4. Env calls seller model via `litellm.acompletion(...)`.
5. Seller action is parsed and applied.
6. Episode ends on max turns, deal, or walk-away.

Important seller safety rules enforced in code:

- Seller cannot accept below `seller_reserve_price`.
- Seller cannot offer below `seller_reserve_price` (offer is clamped).
- Seller API failures trigger fallback response and are recorded in `state["seller_errors"]`.

## Rewards

Equal-weight rubric with 7 rewards:

- `surplus_reward`
- `walkaway_penalty`
- `format_reward`
- `efficiency_bonus`
- `anchoring_reward`
- `no_reveal_penalty`
- `concession_rate_penalty`

See `rewards.py` for exact formulas.

## Run Commands

### 1) Seller Smoke Test (real API call)

```bash
uv run python -m unittest -q test_seller_model_smoke.py
```

Verbose:

```bash
uv run python -m unittest -v test_seller_model_smoke.py
```

If `uv` cache permissions fail:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m unittest -q test_seller_model_smoke.py
```

### 2) Evaluate Buyer with Verifiers

```bash
uv run vf-eval buyer_seller -m openai/gpt-4.1-mini -n 5 -r 1
```

Prime Inference style:

```bash
uv run vf-eval buyer_seller \
  -m openai/gpt-4o \
  -k PRIME_API_KEY \
  -b https://api.pinference.ai/api/v1 \
  -n 1 -r 1 -s
```

### 3) Generate Dataset

```bash
uv run python generate_dataset.py --n 100 --output dataset.json --seed 42
```

Note: `--balanced` is enabled by default in the script.

## Current Sample Dataset (`dataset.json`)

- Episodes: `10`
- Categories: balanced (`2` each across 5 categories)
- Difficulties: `easy/medium/hard/no_deal` mix
- Generator version: `1.1-template`
