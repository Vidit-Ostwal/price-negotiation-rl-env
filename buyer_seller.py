"""
NegotiationEnv — Verifiers MultiTurnEnv for LLM Negotiation RL
================================================================
Buyer  (weak, trainable) — driven by verifiers training loop
Seller (strong, fixed)   — called via litellm inside env_response()

Required env vars:
    OPENAI_API_KEY      — API key for seller model
    SELLER_MODEL        — e.g. "gpt-4o"
    OPENAI_API_BASE     — e.g. "https://api.openai.com/v1"
    DATASET_PATH        — path to dataset.json

Optional:
    MAX_TURNS           — default 10

Usage:
    from negotiation_env import load_environment
    env = load_environment()   # verifiers takes it from here
"""

import logging

from litellm import acompletion
from datasets import Dataset
import verifiers as vf
from verifiers.types import Messages, State
from utils import _validate_env, _build_seller_messages, _load_hf_dataset
from rewards import (
    _parse_action,
    surplus_reward,
    walkaway_penalty,
    format_reward,
    efficiency_bonus,
    anchoring_reward,
    no_reveal_penalty,
    concession_rate_penalty,
)

logger = logging.getLogger(__name__)

MAX_TURNS_DEFAULT = 10


def _init_state(episode: dict) -> dict:
    v = episode["valuations"]
    return {
        "buyer_prompt":     episode["buyer_prompt"],
        "buyer_true_value": v["buyer_true_value"],
        "seller_reserve":   v["seller_reserve_price"],
        "zopa_width":       v["zopa_width"],
        "deal_possible":    v["deal_possible"],
        "seller_prompt":    episode["seller_prompt"],
        "current_offer":    None,
        "offer_history":    [],
        "turn":             0,
        "deal_reached":     False,
        "final_price":      None,
        "walked_away":      False,
        "who_walked":       None,
        "invalid_turns":    0,
    }


# ---------------------------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------------------------

class NegotiationEnv(vf.MultiTurnEnv):

    def __init__(
        self,
        dataset:        Dataset,
        rubric:         vf.Rubric,
        seller_api_key: str,
        seller_model:   str,
        api_base:       str,
        max_turns:      int = MAX_TURNS_DEFAULT,
        **kwargs,
    ):
        super().__init__(dataset=dataset, rubric=rubric, max_turns=max_turns, **kwargs)
        self.seller_api_key = seller_api_key
        self.seller_model   = seller_model
        self.api_base       = api_base

    async def setup_state(self, state: State) -> State:
        episode = state["info"]
        state.update(_init_state(episode))
        state["seller_errors"] = []
        return state

    async def is_completed(self, state: State, **kwargs) -> bool:
        if await super().is_completed(state, **kwargs):
            return True
        return state.get("deal_reached", False) or state.get("walked_away", False)

    def _apply_action(
        self,
        state:            State,
        actor:            str,
        action:           str,
        price:            float | None,
        penalize_invalid: bool,
    ) -> bool:
        """
        Apply a parsed action to state.
        Returns True when the episode should stop immediately.

        For the seller, a hard reserve constraint is enforced
        programmatically — the seller physically cannot accept or
        offer below its reserve price regardless of what the LLM said.
        This ensures deal_possible=False episodes always end without
        a deal, keeping the reward signal clean and trustworthy.
        """
        seller_reserve = state.get("seller_reserve")

        if action == "ACCEPT":
            if state["current_offer"] is not None:

                # Hard constraint: seller cannot accept below reserve
                if actor == "seller" and state["current_offer"] < seller_reserve:
                    logger.debug(
                        f"Seller tried to ACCEPT ${state['current_offer']} "
                        f"below reserve ${seller_reserve} — overriding to WALK"
                    )
                    state["walked_away"] = True
                    state["who_walked"]  = "seller"
                    return True

                state["deal_reached"] = True
                state["final_price"]  = state["current_offer"]
                return True

            if penalize_invalid:
                state["invalid_turns"] += 1
            return False

        if action == "WALK":
            state["walked_away"] = True
            state["who_walked"]  = actor
            return True

        if action == "OFFER" and price is not None:

            # Hard constraint: seller cannot offer below reserve
            if actor == "seller" and price < seller_reserve:
                logger.debug(
                    f"Seller offered ${price} below reserve "
                    f"${seller_reserve} — clamping to reserve"
                )
                price = seller_reserve

            state["current_offer"] = price
            state["offer_history"].append((state["turn"], actor, price))
            return False

        if penalize_invalid:
            state["invalid_turns"] += 1
        return False

    async def env_response(self, messages: Messages, state: State, **kwargs) -> Messages:
        state["turn"] += 1

        # -- Parse buyer -------------------------------------------------
        buyer_action, buyer_price = _parse_action(messages[-1]["content"])
        if self._apply_action(
            state=state,
            actor="buyer",
            action=buyer_action,
            price=buyer_price,
            penalize_invalid=True,
        ):
            return []

        # -- Call seller via litellm -------------------------------------
        seller_messages = _build_seller_messages(messages)
        if not seller_messages or seller_messages[0]["role"] != "user":
            seller_messages.insert(0, {"role": "user", "content": "Begin the negotiation."})

        try:
            response = await acompletion(
                model=self.seller_model,
                messages=[{"role": "system", "content": state["seller_prompt"]}] + seller_messages,
                api_base=self.api_base,
                api_key=self.seller_api_key,
                max_tokens=512,
                temperature=0.7,
            )
            seller_response = response.choices[0].message.content

        except Exception as e:
            err = f"turn={state['turn']} {type(e).__name__}: {e}"
            state["seller_errors"].append(err)
            state["seller_error_last"] = err
            logger.exception("Seller API error; using fallback counter-offer")
            fallback_price = int((state["current_offer"] or 0) * 1.1)
            seller_response = f"I need to hold firm. <action>OFFER ${fallback_price}</action>"

        # -- Parse seller ------------------------------------------------
        seller_action, seller_price = _parse_action(seller_response)
        self._apply_action(
            state=state,
            actor="seller",
            action=seller_action,
            price=seller_price,
            penalize_invalid=False,
        )

        return [{"role": "user", "content": seller_response}]


# ---------------------------------------------------------------------------
# ENTRY POINT — verifiers calls this
# ---------------------------------------------------------------------------

def load_environment() -> NegotiationEnv:
    """
    Validates env vars, loads dataset, returns NegotiationEnv.
    This is the only function verifiers needs.
    """
    config = _validate_env()

    dataset = _load_hf_dataset(config["dataset_path"])
    logger.info(f"✅ Loaded {len(dataset)} episodes from {config['dataset_path']}")

    rubric = vf.Rubric(
        funcs=[
            surplus_reward,
            walkaway_penalty,
            format_reward,
            efficiency_bonus,
            anchoring_reward,
            no_reveal_penalty,
            concession_rate_penalty,
        ],
        weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )

    env = NegotiationEnv(
        dataset=        dataset,
        rubric=         rubric,
        seller_api_key= config["api_key"],
        seller_model=   config["seller_model"],
        api_base=       config["api_base"],
        max_turns=      config["max_turns"],
    )

    logger.info(f"✅ NegotiationEnv ready | seller={config['seller_model']} | max_turns={config['max_turns']}")
    return env
