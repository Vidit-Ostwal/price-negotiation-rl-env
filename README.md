# buyer-seller

Multi-turn buyer/seller negotiation environment for `verifiers`.

- Buyer: policy model selected by `vf-eval -m ...`
- Seller: fixed opponent called inside the environment via LiteLLM

## Repository Layout

- `buyer_seller.py`: `NegotiationEnv` + `load_environment()` entrypoint
- `utils.py`: `.env` loading, env validation, dataset loading, role-flip helper
- `rewards.py`: action parser + 7 reward functions
- `generators/generate_dataset.py`: synthetic dataset generator (template + OpenAI-style LLM + HF chat LLM modes)
- `dataset.json`: sample dataset (10 episodes)
- `test_seller_model_smoke.py`: real seller API smoke test (no mocks)
- `generator.md`: dataset schema + generation guide

## Configuration

The environment uses `utils._validate_env()` and auto-loads `.env` from repo root.

Required:

- `OPENAI_API_KEY`
- `SELLER_MODEL`
- `OPENAI_API_BASE`

Optional:

- `MAX_TURNS` (default `10`)
- `HF_DATASET_REPO` (default `ViditOstwal/price-negotiation-datasets`)
- `HF_DATASET_SPLIT` (default `train`)
- `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) for private HF datasets
- `DATASET_PATH` local fallback path (default `dataset.json`)

Example `.env`:

```bash
OPENAI_API_KEY=sk-...
SELLER_MODEL=openai/gpt-4.1-mini
OPENAI_API_BASE=https://api.openai.com/v1
HF_DATASET_REPO=ViditOstwal/price-negotiation-datasets
HF_DATASET_SPLIT=train
DATASET_PATH=dataset.json
MAX_TURNS=10
```

## Runtime Flow

1. `load_environment()` validates env vars and loads dataset (HF `train` split first, then local `dataset.json` fallback).
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

Rubric with 7 rewards (`surplus_reward` weighted 3x, others 1x):

- `surplus_reward`
- `walkaway_penalty`
- `format_reward`
- `efficiency_bonus`
- `anchoring_reward`
- `concession_rate_reward`
- `decreasing_concessions_reward`

Reward targets:
- `surplus_reward`: maximize buyer value capture on deals.
- `walkaway_penalty`: reward correct outcome decisions (close when feasible, walk when infeasible).
- `format_reward`: keep buyer action tags consistently valid.
- `efficiency_bonus`: finish successful deals in fewer turns.
- `anchoring_reward`: encourage a strong but realistic opening anchor near the ideal point.
- `concession_rate_reward`: discourage large per-turn upward concessions.
- `decreasing_concessions_reward`: encourage concession sequences that shrink over time.

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

Template mode (default):

```bash
uv run python generators/generate_dataset.py --mode template --n 100 --output dataset.json --seed 42
```

LLM mode:

```bash
uv run python generators/generate_dataset.py --mode llm --n 100 --output dataset.json --seed 42
```

HF LLM mode:

```bash
HF_LLM_MODEL=Qwen/Qwen2.5-72B-Instruct:novita \
HF_TOKEN=hf_... \
uv run python generators/generate_dataset.py --mode hf-llm --n 100 --output dataset.json --seed 42
```

LLM mode + push to Hugging Face Hub:

```bash
HF_TOKEN=hf_... \
HF_DATASET_REPO=your-hf-username/price-negotiation-dataset \
uv run python generators/generate_dataset.py \
  --mode llm --n 100 --output dataset.json --seed 42 \
  --push-to-hf --hf-split train
```

LLM mode env vars:

- Required: `OPENAI_API_KEY`
- Optional: `OPENAI_API_BASE` (default `https://api.openai.com/v1`)
- Optional: `GENERATOR_MODEL` (default `gpt-4o-mini`)
- Required for `--mode hf-llm`: `HF_LLM_MODEL`
- Required for `--mode hf-llm`: `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`
- Optional for `--mode hf-llm`: `HF_LLM_API_BASE`
- Optional for HF push: `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`)
- Optional for HF push: `HF_DATASET_REPO` (or `HF_REPO_ID`)

HF write modes:
- `append` (default): load existing split and append newly generated rows before push
- `overwrite`: replace the target split with newly generated rows
- For `--mode llm` or `--mode hf-llm` with `--push-to-hf`, the generator checkpoints by default every `100` rows (`--hf-push-every`) so partial progress is preserved if a later step fails.

`generators/generate_dataset.py` auto-loads missing values from repo-root `.env` before validation.

Note: category balancing is enabled by default in current script behavior.

## Dataset Categories

The generator currently samples from 10 categories:

- `antiques`
- `electronics`
- `collectibles`
- `vehicles`
- `art`
- `furniture`
- `jewelry`
- `musical_instruments`
- `sports_outdoors`
- `luxury_fashion`

In template mode, every category has a curated product bank. In LLM modes, the category is passed to the model and the product is generated dynamically.

## Current Sample Dataset (`dataset.json`)

- Episodes: `10`
- Categories: depends on the file contents; balanced generation now spreads rows across 10 categories
- Difficulties: `easy/medium/hard/no_deal` mix
- Generator version:
  - `1.1-template` for template generation
  - `2.0-llm` for LiteLLM/OpenAI-style generation
  - `2.1-hf-llm` for Hugging Face chat generation
