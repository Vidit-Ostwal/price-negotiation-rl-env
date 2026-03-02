# buyer-seller

Multi-turn negotiation environment built on `verifiers.MultiTurnEnv`.

The buyer is the policy model that `verifiers` evaluates/trains.  
The seller is the environment-side opponent called inside `env_response()` with LiteLLM.

## Overview

- Environment ID: `buyer_seller`
- Task type: multi-turn negotiation
- Domain: buyer/seller price bargaining with hidden private values
- Core objective: maximize buyer-side utility while following strict action formatting

## How The Environment Works

At a high level:

1. `verifiers` asks the buyer model for a response each turn.
2. `NegotiationEnv.env_response()` receives the conversation, parses the buyer action, and updates state.
3. If the negotiation is not terminal, the environment calls the seller model (fixed opponent).
4. Seller response is parsed and state is updated.
5. Loop continues until max turns, deal, or walk-away.
6. Rubric reward functions score the final trajectory.

### Turn Protocol

Both buyer and seller are expected to include one valid action tag in text:

- `<action>OFFER $1234</action>`
- `<action>ACCEPT</action>`
- `<action>WALK</action>`

If buyer formatting is invalid, `invalid_turns` is incremented and `format_reward` penalizes quality.

## Buyer vs Seller Roles

- Buyer model:
  - Chosen by `vf-eval -m <buyer_model>`
  - Called by `verifiers` rollout loop (`get_model_response`)
  - This is the policy being optimized/evaluated
- Seller model:
  - Chosen by `SELLER_MODEL` environment variable
  - Called inside `buyer_seller.py` `env_response()`
  - Treated as fixed environment behavior

## Configuration

This environment reads configuration from environment variables (via `utils._validate_env()`).

| Variable | Required | Example | Used for |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | Yes | `sk-...` | Seller API call auth (and `run_rollout.py` buyer/seller calls) |
| `OPENAI_API_BASE` | Yes | `https://api.openai.com/v1` | OpenAI-compatible endpoint base for LiteLLM |
| `SELLER_MODEL` | Yes | `openai/gpt-4.1-mini` | Seller model ID used in environment turn responses |
| `DATASET_PATH` | Yes | `dataset.json` | Episode dataset path |
| `BUYER_MODEL` | Yes (current validator) | `openai/gpt-4.1-mini` | Used by `run_rollout.py` (not used directly by `vf-eval` buyer call) |
| `MAX_TURNS` | No | `10` | Max negotiation turns (default: `10`) |

### Important Note About `BUYER_MODEL`

For `vf-eval`, buyer model selection comes from the CLI `-m` flag.  
However, current env validation still requires `BUYER_MODEL` to be set because the same validator is shared with `run_rollout.py`.

## Dataset Format

`DATASET_PATH` should point to a JSON list of episodes. Each episode is expected to include:

- `buyer_prompt` (str): system prompt shown to buyer model
- `seller_prompt` (str): system prompt used for seller model call
- `valuations` (dict), including:
  - `buyer_true_value`
  - `seller_reserve_price`
  - `zopa_width`
  - `deal_possible`
- Additional metadata fields are preserved in `info`

At load time, dataset rows are converted to:

- `prompt`: `[{"role": "system", "content": buyer_prompt}]` (chat format)
- `info`: full episode object

## State Fields

Per-episode state is initialized in `setup_state()` using `_init_state(...)`:

- `buyer_prompt`, `seller_prompt`
- `buyer_true_value`, `seller_reserve`, `zopa_width`, `deal_possible`
- `current_offer`, `offer_history`, `turn`
- `deal_reached`, `final_price`, `walked_away`, `who_walked`
- `invalid_turns`

These are mutated in-place during `env_response()`.

## Rewards

Defined in `rewards.py` and combined with equal weights in a rubric:

1. `surplus_reward`
- Buyer surplus on successful deals:
  - hard `-1.0` if a deal closes above `buyer_true_value`
  - otherwise `(buyer_true_value - final_price) / zopa_width` clamped to `[-1, 1]`
  - returns `0.0` if no deal

2. `walkaway_penalty`
- If no deal:
  - `-0.5` when a deal was possible
  - `+0.2` when no deal was possible
- returns `0.0` when deal reached

3. `format_reward`
- Fraction of buyer turns with valid action tags, scaled by `0.2`
- max contribution `0.2`

4. `efficiency_bonus`
- Faster successful closures get higher bonus:
  - `((max_turns - turns_used) / max_turns) * 0.1`
- max contribution `0.1`

5. `anchoring_reward`
- Uses hidden `suggested_buyer_anchor` from episode info:
  - opening below anchor: `+0.3`
  - opening at anchor: `+0.15`
  - opening above anchor: `-0.2`

6. `no_reveal_penalty`
- Penalizes revealing buyer ceiling too early:
  - any buyer offer >= 90% of `buyer_true_value` -> `-0.3`

7. `concession_rate_penalty`
- Penalizes overly fast upward concessions:
  - if average concession pace is too high, penalty up to `-0.3`

## Running With `vf-eval`

Example:

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_API_BASE=https://api.openai.com/v1
export BUYER_MODEL=openai/gpt-4.1-mini
export SELLER_MODEL=openai/gpt-4.1-mini
export DATASET_PATH=dataset.json
export MAX_TURNS=10

uv run vf-eval buyer_seller -m openai/gpt-4.1-mini -n 5 -r 1
```

Prime Inference-style call:

```bash
uv run vf-eval buyer_seller \
  -m openai/gpt-4o \
  -k PRIME_API_KEY \
  -b https://api.pinference.ai/api/v1 \
  -n 1 -r 1 -s
```

What happens:

- `-m` sets the buyer model used by verifiers rollout.
- `SELLER_MODEL` sets the fixed seller model called by the environment.
- `-k`/`-b` configure the buyer-side inference endpoint for `vf-eval`.

## Local Rollout Validation Script

`run_rollout.py` is a standalone validator that calls both buyer and seller via LiteLLM (outside the verifiers training loop).

Run:

```bash
uv run run_rollout.py --episodes 5 --concurrency 2 --verbose
```

Output:

- Per-episode transcript and actions
- Reward breakdown (`surplus`, `walkaway`, `format`, `efficiency`, `total`)
- Aggregated summary
- JSON dump to `rollout_results.json` by default

## Module Structure

- `buyer_seller.py`: environment class and `load_environment()`
- `rewards.py`: action parser + reward functions
- `utils.py`: env var validation, dataset loader, seller-message role conversion
- `run_rollout.py`: standalone two-model rollout checker
- `generate_dataset.py`: dataset generation utility

## Known Constraints

- Current `_validate_env()` requires `BUYER_MODEL` even for pure `vf-eval` flow.
- Seller API fallback behavior on errors is a hard-coded counter-offer strategy.
- Environment assumes episodes contain expected valuation keys.
