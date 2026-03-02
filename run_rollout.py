"""
run_rollout.py — Validation Script
====================================
Runs negotiation episodes with BOTH buyer and seller as litellm API calls.
Purpose: validate the rollout logic works before plugging into verifiers training.

This file is NOT part of the training pipeline.
It's purely to inspect transcripts and verify reward signals are correct.

Required env vars:
    OPENAI_API_KEY      — API key
    BUYER_MODEL         — weak model  e.g. "gpt-4o-mini"
    SELLER_MODEL        — strong model e.g. "gpt-4o"
    OPENAI_API_BASE     — e.g. "https://api.openai.com/v1"
    DATASET_PATH        — path to dataset.json

Usage:
    python run_rollout.py --episodes 3 --verbose
"""

import os
import sys
import json
import asyncio
import argparse
import logging
from typing import Optional

import litellm
from litellm import acompletion

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Silence LiteLLM internal logs; keep this script's logs readable.
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("litellm").setLevel(logging.CRITICAL)

# ---------------------------------------------------------------------------
# IMPORT ENV HELPERS FROM negotiation_env
# (reuse the same action parser, state, rewards — don't duplicate)
# ---------------------------------------------------------------------------

from buyer_seller import (
    _parse_action,
    _init_state,
)
from rewards import (
    surplus_reward,
    walkaway_penalty,
    format_reward,
    efficiency_bonus,
)
from utils import _validate_env

# ---------------------------------------------------------------------------
# CALL MODEL
# ---------------------------------------------------------------------------

async def _call_model(
    model:         str,
    system_prompt: str,
    messages:      list[dict],
    api_key:       str,
    api_base:      str,
) -> str:
    response = await acompletion(
        model=model,
        messages=[{"role": "system", "content": system_prompt}] + messages,
        api_base=api_base,
        api_key=api_key,
        max_tokens=512,
        temperature=0.7,
    )
    return response.choices[0].message.content

# ---------------------------------------------------------------------------
# SINGLE EPISODE
# ---------------------------------------------------------------------------

async def run_episode(
    episode:      dict,
    buyer_model:  str,
    seller_model: str,
    api_key:      str,
    api_base:     str,
    max_turns:    int = 10,
    verbose:      bool = False,
) -> dict:
    """
    Runs one full negotiation episode.
    Buyer and seller alternate turns as litellm API calls.
    Returns result dict with transcript + rewards.
    """
    state = _init_state(episode)

    # Separate message histories — each agent sees the conversation from its own POV
    buyer_history  = []   # buyer=assistant,  seller=user
    seller_history = []   # seller=assistant, buyer=user

    transcript = []

    if verbose:
        v = episode["valuations"]
        print(f"\n{'='*60}")
        print(f"  {episode['product']['name']}")
        print(f"  Buyer value : ${v['buyer_true_value']}  |  Seller reserve: ${v['seller_reserve_price']}")
        print(f"  ZOPA        : {v['zopa']}  |  Difficulty: {v['difficulty']}")
        print(f"{'='*60}")

    while state["turn"] < max_turns:
        state["turn"] += 1

        # ── BUYER TURN ───────────────────────────────────────────
        try:
            buyer_msg = await _call_model(
                model=buyer_model,
                system_prompt=state["buyer_prompt"],
                messages=buyer_history,
                api_key=api_key,
                api_base=api_base,
            )
        except Exception as e:
            logger.error(f"Buyer API error at turn {state['turn']}: {e}")
            break

        buyer_action, buyer_price = _parse_action(buyer_msg)
        transcript.append({"turn": state["turn"], "role": "buyer",
                            "action": buyer_action, "price": buyer_price, "message": buyer_msg})

        if verbose:
            price_str = f" ${int(buyer_price)}" if buyer_price else ""
            print(f"\n  [T{state['turn']}] BUYER  ({buyer_model})")
            print(f"  Action : {buyer_action}{price_str}")
            print(f"  Says   : {buyer_msg[:300]}{'...' if len(buyer_msg) > 300 else ''}")

        # Update both histories
        buyer_history.append( {"role": "assistant", "content": buyer_msg})
        seller_history.append({"role": "user",      "content": buyer_msg})

        # Check buyer terminal
        if buyer_action == "ACCEPT" and state["current_offer"] is not None:
            state["deal_reached"] = True
            state["final_price"]  = state["current_offer"]
            if verbose: print(f"\n  ✅ DEAL — buyer accepted ${state['final_price']}")
            break

        if buyer_action == "WALK":
            state["walked_away"] = True
            state["who_walked"]  = "buyer"
            if verbose: print(f"\n  🚶 Buyer walked away")
            break

        if buyer_action == "OFFER" and buyer_price is not None:
            state["current_offer"] = buyer_price
            state["offer_history"].append((state["turn"], "buyer", buyer_price))
        else:
            state["invalid_turns"] += 1

        # ── SELLER TURN ──────────────────────────────────────────
        try:
            seller_msg = await _call_model(
                model=seller_model,
                system_prompt=state["seller_prompt"],
                messages=seller_history,
                api_key=api_key,
                api_base=api_base,
            )
        except Exception as e:
            logger.error(f"Seller API error at turn {state['turn']}: {e}")
            break

        seller_action, seller_price = _parse_action(seller_msg)
        transcript.append({"turn": state["turn"], "role": "seller",
                            "action": seller_action, "price": seller_price, "message": seller_msg})

        if verbose:
            price_str = f" ${int(seller_price)}" if seller_price else ""
            print(f"\n  [T{state['turn']}] SELLER ({seller_model})")
            print(f"  Action : {seller_action}{price_str}")
            print(f"  Says   : {seller_msg[:300]}{'...' if len(seller_msg) > 300 else ''}")

        # Update both histories
        seller_history.append({"role": "assistant", "content": seller_msg})
        buyer_history.append( {"role": "user",      "content": seller_msg})

        # Check seller terminal
        if seller_action == "ACCEPT" and state["current_offer"] is not None:
            state["deal_reached"] = True
            state["final_price"]  = state["current_offer"]
            if verbose: print(f"\n  ✅ DEAL — seller accepted ${state['final_price']}")
            break

        if seller_action == "WALK":
            state["walked_away"] = True
            state["who_walked"]  = "seller"
            if verbose: print(f"\n  🚶 Seller walked away")
            break

        if seller_action == "OFFER" and seller_price is not None:
            state["current_offer"] = seller_price
            state["offer_history"].append((state["turn"], "seller", seller_price))

    else:
        if verbose: print(f"\n  ⏰ Max turns reached — no deal")

    # Compute rewards
    rewards = {
        "surplus":    await surplus_reward(transcript,   episode, state=state),
        "walkaway":   await walkaway_penalty(transcript, episode, state=state),
        "format":     await format_reward(transcript,    episode, state=state),
        "efficiency": await efficiency_bonus(transcript, episode, state=state),
    }
    rewards["total"] = sum(rewards.values())

    if verbose:
        print(f"\n  {'─'*40}")
        print(f"  Rewards → surplus:{rewards['surplus']:.2f}  walkaway:{rewards['walkaway']:.2f}  "
              f"format:{rewards['format']:.2f}  efficiency:{rewards['efficiency']:.2f}  "
              f"TOTAL:{rewards['total']:.2f}")

    return {
        "episode_id":       episode["episode_id"],
        "product":          episode["product"]["name"],
        "difficulty":       episode["valuations"]["difficulty"],
        "buyer_true_value": episode["valuations"]["buyer_true_value"],
        "seller_reserve":   episode["valuations"]["seller_reserve_price"],
        "deal_reached":     state["deal_reached"],
        "final_price":      state["final_price"],
        "walked_away":      state["walked_away"],
        "who_walked":       state["who_walked"],
        "turns_used":       state["turn"],
        "rewards":          rewards,
        "transcript":       transcript,
    }

# ---------------------------------------------------------------------------
# BATCH
# ---------------------------------------------------------------------------

async def run_rollouts(
    n_episodes:  int = 5,
    concurrency: int = 2,
    verbose:     bool = False,
    output_path: Optional[str] = "rollout_results.json",
):
    config = _validate_env()

    with open(config["dataset_path"]) as f:
        episodes = json.load(f)
    episodes = episodes[:n_episodes]

    print(f"\n🎰 Running {len(episodes)} episodes")
    print(f"   buyer  : {config['buyer_model']}")
    print(f"   seller : {config['seller_model']}\n")

    sem = asyncio.Semaphore(concurrency)

    async def run_with_sem(ep, idx):
        async with sem:
            if not verbose:
                logger.info(f"[{idx+1}/{len(episodes)}] {ep['product']['name']} ({ep['valuations']['difficulty']})")
            return await run_episode(
                episode=      ep,
                buyer_model=  config["buyer_model"],
                seller_model= config["seller_model"],
                api_key=      config["api_key"],
                api_base=     config["api_base"],
                max_turns=    config["max_turns"],
                verbose=      verbose,
            )

    results = await asyncio.gather(*[run_with_sem(ep, i) for i, ep in enumerate(episodes)])
    results = [r for r in results if r]

    # Summary
    deals    = [r for r in results if r["deal_reached"]]
    avg_r    = sum(r["rewards"]["total"] for r in results) / len(results)
    avg_surp = sum(r["rewards"]["surplus"] for r in deals) / len(deals) if deals else 0

    print(f"\n{'='*50}")
    print(f"  Episodes : {len(results)}")
    print(f"  Deals    : {len(deals)} ({100*len(deals)/len(results):.0f}%)")
    print(f"  Avg reward  : {avg_r:.3f}")
    print(f"  Avg surplus : {avg_surp:.3f}  (deals only)")
    print(f"{'='*50}\n")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"💾 Saved to {output_path}")

    return results

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate negotiation rollouts")
    parser.add_argument("--episodes",    type=int,  default=5)
    parser.add_argument("--concurrency", type=int,  default=2)
    parser.add_argument("--output",      default="rollout_results.json")
    parser.add_argument("--verbose",     action="store_true", help="Print full transcripts")
    args = parser.parse_args()

    asyncio.run(run_rollouts(
        n_episodes=  args.episodes,
        concurrency= args.concurrency,
        verbose=     args.verbose,
        output_path= args.output,
    ))
