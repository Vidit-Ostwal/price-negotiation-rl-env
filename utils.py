import json
import os
import sys
from pathlib import Path

from datasets import Dataset
from verifiers.types import Messages

MAX_TURNS_DEFAULT = 10

REQUIRED_ENV_VARS = {
    "OPENAI_API_KEY": "API key for seller model",
    "SELLER_MODEL": "Strong seller model (e.g. gpt-4o)",
    "OPENAI_API_BASE": "OpenAI-compatible base URL",
    "DATASET_PATH": "Path to dataset.json",
}


def _load_dotenv(dotenv_path: str = ".env") -> None:
    """
    Load KEY=VALUE pairs from a .env file into process env if keys are unset.
    Keeps explicit shell exports higher priority than .env values.
    """
    env_file = Path(dotenv_path)
    if not env_file.exists():
        return

    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def _validate_env() -> dict:
    """
    Check all required env vars are present.
    Prints a clear error for every missing key and exits if any are absent.
    Returns a clean config dict if all are present.
    """
    _load_dotenv()

    missing = [
        (var, desc)
        for var, desc in REQUIRED_ENV_VARS.items()
        if not os.getenv(var)
    ]

    if missing:
        print("\n❌  Missing required environment variables:\n")
        for var, desc in missing:
            print(f"    {var:<22}  # {desc}")
        print(
            "\nExample:\n"
            "    # Add values to .env (recommended), or export manually:\n"
            "    export OPENAI_API_KEY=sk-...\n"
            "    export SELLER_MODEL=gpt-4o\n"
            "    export OPENAI_API_BASE=https://api.openai.com/v1\n"
            "    export DATASET_PATH=dataset.json\n"
        )
        sys.exit(1)

    return {
        "api_key": os.environ["OPENAI_API_KEY"],
        "seller_model": os.environ["SELLER_MODEL"],
        "api_base": os.environ["OPENAI_API_BASE"],
        "dataset_path": os.environ["DATASET_PATH"],
        "max_turns": int(os.getenv("MAX_TURNS", str(MAX_TURNS_DEFAULT))),
    }


def _build_seller_messages(conversation: Messages) -> list[dict]:
    """Flip roles so seller sees itself as assistant."""
    seller_messages = []
    for msg in conversation:
        if msg["role"] == "system":
            continue
        if msg["role"] == "assistant":
            seller_messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "user":
            seller_messages.append({"role": "assistant", "content": msg["content"]})
    return seller_messages


def _load_hf_dataset(dataset_path: str) -> Dataset:
    with open(dataset_path) as f:
        episodes = json.load(f)
    return Dataset.from_list([
        {
            "prompt": [{"role": "system", "content": ep["buyer_prompt"]}],
            "info": ep,
        }
        for ep in episodes
    ])
