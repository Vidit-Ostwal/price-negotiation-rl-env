import re
from typing import Optional

from verifiers.types import Messages
from utils import MAX_TURNS_DEFAULT

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
# REWARDS
# ---------------------------------------------------------------------------

async def surplus_reward(completion: Messages, info: dict, **kwargs) -> float:
    """
    Reward buyer surplus on successful deals.

    The primary outcome reward — how much value did the buyer capture
    relative to what was available in the ZOPA?

    Behavior:
    - Returns 0.0 if no deal was reached.
    - Returns -1.0 if deal closes above buyer_true_value (hard overpay).
    - Otherwise: (buyer_true_value - final_price) / zopa_width
      +1.0 = closed at seller reserve (captured full ZOPA)
       0.0 = closed at buyer true value (broke even)
      -1.0 = overpaid above true value

    State keys: deal_reached, buyer_true_value, final_price, zopa_width
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
    Reward/penalty for whether the deal outcome was correct.

    With the hard seller reserve constraint in the environment,
    deal_possible=False + deal_reached=True is now impossible.
    That case is kept as a safety net but should never fire.

    Behavior:
    - deal_possible=True,  deal_reached=True  -> +1.0  correct deal
    - deal_possible=True,  deal_reached=False -> -1.0  missed a deal
    - deal_possible=False, deal_reached=False -> +1.0  correct walk
    - deal_possible=False, deal_reached=True  ->  0.0  should not happen

    State keys: deal_reached, deal_possible
    """
    state = kwargs.get("state", {})
    deal_reached  = bool(state.get("deal_reached", False))
    deal_possible = bool(state.get("deal_possible", True))

    if deal_possible and deal_reached:
        return 1.0
    if deal_possible and not deal_reached:
        return -1.0
    if not deal_possible and not deal_reached:
        return 1.0
    # deal_possible=False, deal_reached=True — impossible with hard constraint
    return 0.0


async def format_reward(completion: Messages, info: dict, **kwargs) -> float:
    """
    Reward compliance with required buyer action-tag formatting.

    Continuous fraction of valid buyer turns. Already a clean
    continuous signal — no changes needed.

    Behavior:
    - Returns valid_turns / total_buyer_turns in [0.0, 1.0].

    Inputs: completion message list
    """
    buyer_turns = [m for m in completion if m["role"] == "assistant"]
    if not buyer_turns:
        return 0.0
    valid = sum(1 for m in buyer_turns if _parse_action(m["content"])[0] != "INVALID")
    return valid / len(buyer_turns)


async def efficiency_bonus(completion: Messages, info: dict, **kwargs) -> float:
    """
    Reward closing a deal in fewer turns.

    Continuous fraction of turns saved. Already a clean signal.

    Behavior:
    - Returns 0.0 if no deal.
    - Returns (max_turns - turns_used) / max_turns in [0.0, 1.0].

    State keys: deal_reached, turn, max_turns (optional)
    Info keys:  metadata.max_turns (optional)
    """
    state = kwargs.get("state", {})
    if not state.get("deal_reached"):
        return 0.0
    max_turns = (
        state.get("max_turns")
        or info.get("metadata", {}).get("max_turns")
        or MAX_TURNS_DEFAULT
    )
    return (max_turns - state.get("turn", max_turns)) / max_turns


async def anchoring_reward(completion: Messages, info: dict, **kwargs) -> float:
    """
    Reward opening with a strategically low anchor.

    The model is NOT told where to open — it must discover through
    interaction that opening low leads to better outcomes.
    suggested_buyer_anchor (stored in episode info) is used only
    here as an evaluation threshold, never shown to the buyer.

    Continuous signal centered on the ideal anchor (65% of true value).
    Smooth decay as the opening offer moves away from ideal in either
    direction — no arbitrary band, no cliff.

    Formula:
        ideal    = 0.65 * buyer_true_value
        distance = |opening_offer - ideal| / buyer_true_value
        score    = 1.0 - 2.0 * distance

    Examples (buyer_true_value = $1000, ideal = $650):
        open $650 -> score +1.0   (perfect anchor)
        open $750 -> score +0.8   (acceptable)
        open $850 -> score +0.6   (too high but not catastrophic)
        open $950 -> score +0.4   (very high)
        open $550 -> score +0.8   (slightly below ideal, fine)
        open $400 -> score +0.5   (too low, risks insulting seller)

    State keys: buyer_true_value
    Completion: first buyer OFFER
    """
    state = kwargs.get("state", {})
    buyer_true_value = (
        state.get("buyer_true_value")
        or info.get("valuations", {}).get("buyer_true_value")
    )
    if not buyer_true_value:
        return 0.0

    buyer_offers = _get_buyer_offers(completion)
    if not buyer_offers:
        return 0.0

    opening_offer = buyer_offers[0]
    ideal         = 0.65 * buyer_true_value
    distance      = abs(opening_offer - ideal) / buyer_true_value
    return float(1.0 - 2.0 * distance)


async def concession_rate_reward(completion: Messages, info: dict, **kwargs) -> float:
    """
    Reward making small concessions per turn.

    A buyer who makes large upward jumps signals desperation —
    the seller holds firm expecting more. Strategic buyers concede
    in small, controlled steps.

    Fully continuous — no cliff, no dead zone. The gradient exists
    everywhere so the model can always improve.

    Formula:
        concessions      = upward moves between consecutive buyer offers
        avg_concession   = mean(concessions)
        concession_ratio = avg_concession / buyer_true_value
        score            = 1.0 - 4.0 * concession_ratio

    Examples (buyer_true_value = $1000):
        avg concession $0   -> +1.0  (held firm, ideal)
        avg concession $25  -> +0.9  (very controlled)
        avg concession $50  -> +0.8  (good)
        avg concession $150 -> +0.4  (getting large)
        avg concession $250 ->  0.0  (break-even)
        avg concession $350 -> -0.4  (bad, signaling desperation)

    State keys: buyer_true_value
    Completion: all buyer OFFER messages in order
    """
    state = kwargs.get("state", {})
    buyer_true_value = state.get("buyer_true_value")
    if not buyer_true_value:
        return 0.0

    buyer_offers = _get_buyer_offers(completion)
    if len(buyer_offers) < 2:
        return 0.0

    concessions = [
        max(0.0, buyer_offers[i] - buyer_offers[i - 1])
        for i in range(1, len(buyer_offers))
    ]

    avg_concession   = sum(concessions) / len(concessions)
    concession_ratio = avg_concession / buyer_true_value
    return float(1.0 - 4.0 * concession_ratio)


async def decreasing_concessions_reward(completion: Messages, info: dict, **kwargs) -> float:
    """
    Reward a decreasing pattern of concessions over time.

    Good negotiators signal they are approaching their limit by making
    each successive concession smaller than the last. This is distinct
    from concession_rate_reward which only cares about magnitude —
    this rewards the *shape* of the concession curve.

    Requires at least 3 buyer offers (2 concession steps) to compute
    a meaningful pattern. Returns 0.0 with fewer offers.

    Method:
        1. Extract all upward concession steps between consecutive offers.
        2. For each adjacent pair of steps, check if later <= earlier.
        3. Score = fraction of consecutive pairs that are non-increasing.
           +1.0 = perfectly decreasing pattern throughout
            0.0 = random or flat pattern
           -1.0 not possible (score is in [0.0, 1.0])

    Then center the score around 0 so it can penalize bad patterns:
        final_score = 2.0 * fraction - 1.0
        +1.0 = fully decreasing (all steps shrinking)
         0.0 = half decreasing, half increasing (random)
        -1.0 = fully increasing (getting worse each turn)

    Examples:
        concessions [200, 100, 50]  -> fraction 1.0 -> score +1.0
        concessions [100, 100, 100] -> fraction 1.0 -> score +1.0  (flat = ok)
        concessions [50, 100, 200]  -> fraction 0.0 -> score -1.0
        concessions [200, 100, 200] -> fraction 0.5 -> score  0.0

    State keys: (none)
    Completion: all buyer OFFER messages in order
    """
    buyer_offers = _get_buyer_offers(completion)
    if len(buyer_offers) < 3:
        return 0.0

    concessions = [
        max(0.0, buyer_offers[i] - buyer_offers[i - 1])
        for i in range(1, len(buyer_offers))
    ]

    # Need at least 2 concession steps to compare a pattern
    if len(concessions) < 2:
        return 0.0

    non_increasing = sum(
        1 for i in range(1, len(concessions))
        if concessions[i] <= concessions[i - 1]
    )

    fraction = non_increasing / (len(concessions) - 1)
    return float(2.0 * fraction - 1.0)