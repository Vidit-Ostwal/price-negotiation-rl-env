import re
from typing import Optional

from verifiers.types import Messages

DEFAULT_MAX_TURNS = 10

ACTION_RE = re.compile(
    r"<action>(OFFER\s*\$?([\d,]+(?:\.\d+)?)|ACCEPT|WALK)</action>",
    re.IGNORECASE,
)


def _parse_action(message: str) -> tuple[str, Optional[float]]:
    match = ACTION_RE.search(message)
    if not match:
        return ("INVALID", None)
    raw = match.group(1).upper().strip()
    if raw == "ACCEPT":
        return ("ACCEPT", None)
    if raw == "WALK":
        return ("WALK", None)
    if raw.startswith("OFFER"):
        price_str = match.group(2)
        if price_str:
            return ("OFFER", float(price_str.replace(",", "")))
    return ("INVALID", None)


def _get_buyer_offers(completion: Messages) -> list[float]:
    """Extract ordered list of prices from buyer OFFER turns."""
    offers = []
    for m in completion:
        if m["role"] == "assistant":
            action, price = _parse_action(m["content"])
            if action == "OFFER" and price is not None:
                offers.append(price)
    return offers


# ---------------------------------------------------------------------------
# ORIGINAL REWARDS
# ---------------------------------------------------------------------------

async def surplus_reward(completion: Messages, info: dict, **kwargs) -> float:
    """
    Reward buyer surplus on successful deals.

    Behavior:
    - Returns 0.0 if no deal was reached.
    - Returns -1.0 if deal closes above buyer_true_value (hard overpay penalty).
    - Otherwise computes normalized buyer surplus:
      (buyer_true_value - final_price) / zopa_width
    - Positive when buyer closes below their true value.
    - Negative when buyer overpays above true value.
    - Clamps final value to [-1.0, 1.0].

    State keys used:
    - deal_reached, buyer_true_value, final_price, zopa_width
    """
    state = kwargs.get("state", {})
    if not state.get("deal_reached"):
        return 0.0
    if state["final_price"] > state["buyer_true_value"]:
        return -1.0
    zopa_width = state["zopa_width"]
    if zopa_width <= 0:
        return 0.0
    surplus = (state["buyer_true_value"] - state["final_price"]) / zopa_width
    return float(max(-1.0, min(1.0, surplus)))


async def walkaway_penalty(completion: Messages, info: dict, **kwargs) -> float:
    """
    Reward/penalty for whether ending without a deal was appropriate.

    Behavior:
    - Returns 0.0 if a deal was reached.
    - Returns -0.5 if no deal was reached but a deal was possible.
    - Returns +0.2 if no deal was reached and no deal was possible.

    State keys used:
    - deal_reached, deal_possible
    """
    state = kwargs.get("state", {})
    if state.get("deal_reached"):
        return 0.0
    return -0.5 if state.get("deal_possible", True) else 0.2


async def format_reward(completion: Messages, info: dict, **kwargs) -> float:
    """
    Reward compliance with required buyer action-tag formatting.

    Behavior:
    - Looks only at buyer messages (role == "assistant").
    - Checks each buyer message for a valid <action> tag:
      OFFER, ACCEPT, or WALK.
    - Returns (valid_turn_fraction * 0.2), so max contribution is 0.2.

    Inputs used:
    - completion message list
    """
    buyer_turns = [m for m in completion if m["role"] == "assistant"]
    if not buyer_turns:
        return 0.0
    valid = sum(1 for m in buyer_turns if _parse_action(m["content"])[0] != "INVALID")
    return (valid / len(buyer_turns)) * 0.2


async def efficiency_bonus(completion: Messages, info: dict, **kwargs) -> float:
    """
    Small bonus for closing a deal in fewer turns.

    Behavior:
    - Returns 0.0 if no deal was reached.
    - Uses max_turns from info["metadata"]["max_turns"] when present,
      otherwise falls back to DEFAULT_MAX_TURNS.
    - Computes ((max_turns - turns_used) / max_turns) * 0.1.
    - Max contribution is 0.1.

    State/info keys used:
    - state.deal_reached, state.turn
    - info.metadata.max_turns (optional)
    """
    state = kwargs.get("state", {})
    if not state.get("deal_reached"):
        return 0.0
    max_turns = info.get("metadata", {}).get("max_turns", DEFAULT_MAX_TURNS)
    return ((max_turns - state.get("turn", max_turns)) / max_turns) * 0.1


# ---------------------------------------------------------------------------
# NEW REWARDS
# ---------------------------------------------------------------------------

async def anchoring_reward(completion: Messages, info: dict, **kwargs) -> float:
    """
    Reward the buyer for discovering that opening low is a good strategy.

    The model is NOT told what to open at — it must learn through interaction
    that opening below the suggested_buyer_anchor leads to better outcomes.
    suggested_buyer_anchor is used only here as an evaluation threshold,
    never shown to the buyer during the episode.

    Behavior:
    - Returns 0.0 if buyer made no offers.
    - Compares opening offer to suggested_buyer_anchor from the episode info.
    - Opening BELOW anchor → +0.3  (model discovered good anchoring)
    - Opening EQUAL to anchor → +0.15 (acceptable)
    - Opening ABOVE anchor → -0.2  (model opened too high, bad anchor)

    State keys used:
    - (none — uses info directly)

    Info keys used:
    - info["valuations"]["suggested_buyer_anchor"]

    Completion used:
    - First buyer OFFER message
    """
    suggested_anchor = info.get("valuations", {}).get("suggested_buyer_anchor")
    if not suggested_anchor:
        return 0.0

    buyer_offers = _get_buyer_offers(completion)
    if not buyer_offers:
        return 0.0

    opening_offer = buyer_offers[0]

    if opening_offer < suggested_anchor:
        return 0.3    # model discovered it should open low — reward it
    elif opening_offer == suggested_anchor:
        return 0.15   # exactly at the threshold — acceptable
    else:
        return -0.2   # opened too high — bad anchoring


async def no_reveal_penalty(completion: Messages, info: dict, **kwargs) -> float:
    """
    Penalize the buyer for revealing their true value ceiling.

    If any buyer offer exceeds 90% of buyer_true_value, the buyer has
    essentially shown the seller their ceiling — the seller now knows
    exactly where to anchor their counteroffer.

    Behavior:
    - Returns 0.0 if all buyer offers stay below 90% of true value.
    - Returns -0.3 if any single offer breaches the 90% threshold.
    - Penalty is flat (not scaled) — revealing your ceiling is always bad
      regardless of how close to 100% you go.

    State keys used:
    - buyer_true_value

    Completion used:
    - All buyer OFFER messages
    """
    state = kwargs.get("state", {})
    buyer_true_value = state.get("buyer_true_value")
    if not buyer_true_value:
        return 0.0

    buyer_offers = _get_buyer_offers(completion)
    if not buyer_offers:
        return 0.0

    ceiling = buyer_true_value * 0.90
    if any(offer >= ceiling for offer in buyer_offers):
        return -0.3

    return 0.0


async def concession_rate_penalty(completion: Messages, info: dict, **kwargs) -> float:
    """
    Penalize the buyer for conceding too much too fast.

    A buyer who makes large upward jumps turn-over-turn signals desperation
    to the seller, who will then hold firm expecting further concessions.
    Strategic negotiators make small, decreasing concessions.

    Behavior:
    - Returns 0.0 if fewer than 2 buyer offers (not enough data).
    - Computes the average upward concession per turn as a fraction of
      buyer_true_value:
        avg_concession = mean of max(0, offer[i] - offer[i-1])
        concession_ratio = avg_concession / buyer_true_value
    - If concession_ratio > 0.10 (buyer jumps >10% of their value per turn):
        penalty = -concession_ratio * 0.5, clamped to [-0.3, 0.0]
    - If concession_ratio <= 0.10: no penalty (concessions are controlled).

    State keys used:
    - buyer_true_value

    Completion used:
    - All buyer OFFER messages in order
    """
    state = kwargs.get("state", {})
    buyer_true_value = state.get("buyer_true_value")
    if not buyer_true_value:
        return 0.0

    buyer_offers = _get_buyer_offers(completion)
    if len(buyer_offers) < 2:
        return 0.0

    # Only count upward moves (concessions) — downward moves are fine
    concessions = [
        max(0.0, buyer_offers[i] - buyer_offers[i - 1])
        for i in range(1, len(buyer_offers))
    ]

    avg_concession   = sum(concessions) / len(concessions)
    concession_ratio = avg_concession / buyer_true_value

    if concession_ratio > 0.10:
        penalty = -concession_ratio * 0.5
        return float(max(-0.3, penalty))

    return 0.0
