import json
import unittest
from pathlib import Path

from buyer_seller import load_environment, _init_state
from rewards import _parse_action
from utils import _validate_env


def _load_first_episode(dataset_path: str) -> dict:
    episodes = json.loads(Path(dataset_path).read_text())
    if not episodes:
        raise AssertionError(f"Dataset at {dataset_path} is empty")
    return episodes[0]


class SellerModelSmokeTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = _validate_env()
        cls.first_episode = _load_first_episode(cls.config["dataset_path"])

    def _make_env_and_state(self):
        env = load_environment()
        state = _init_state(self.first_episode)
        state["seller_errors"] = []
        return env, state

    def test_environment_uses_validated_config(self) -> None:
        env = load_environment()
        self.assertEqual(env.seller_api_key, self.config["api_key"])
        self.assertEqual(env.seller_model, self.config["seller_model"])
        self.assertEqual(env.api_base, self.config["api_base"])

    async def test_seller_model_real_api_smoke(self) -> None:
        env, state = self._make_env_and_state()

        # Buyer opens with a valid offer, then env should call the configured seller model.
        messages = [
            {"role": "system", "content": state["buyer_prompt"]},
            {"role": "assistant", "content": "I can start at this. <action>OFFER $900</action>"},
        ]

        seller_turn = await env.env_response(messages, state)

        self.assertEqual(state["turn"], 1)
        self.assertEqual(len(state["seller_errors"]), 0, f"Seller API error: {state.get('seller_error_last')}")
        self.assertEqual(len(seller_turn), 1)
        self.assertEqual(seller_turn[0]["role"], "user")

        seller_action, seller_price = _parse_action(seller_turn[0]["content"])
        self.assertIn(seller_action, {"OFFER", "ACCEPT", "WALK"})

        if seller_action == "OFFER":
            self.assertIsNotNone(seller_price)
            self.assertEqual(state["offer_history"][-1][1], "seller")
            self.assertEqual(state["current_offer"], max(seller_price, state["seller_reserve"]))


if __name__ == "__main__":
    unittest.main()
