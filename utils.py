import json
import os
import sys
from pathlib import Path

from datasets import Dataset, load_dataset
from verifiers.types import Messages

MAX_TURNS_DEFAULT = 10
DEFAULT_DATASET_PATH = "dataset.json"
DEFAULT_HF_DATASET_REPO = "ViditOstwal/price-negotiation-datasets"
DEFAULT_HF_DATASET_SPLIT = "train"

REQUIRED_ENV_VARS = {
    "OPENAI_API_KEY": "API key for seller model",
    "SELLER_MODEL": "Strong seller model (e.g. gpt-4o)",
    "OPENAI_API_BASE": "OpenAI-compatible base URL",
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
            "    export HF_DATASET_REPO=ViditOstwal/price-negotiation-datasets\n"
            "    export DATASET_PATH=dataset.json\n"
        )
        sys.exit(1)

    return {
        "api_key": os.environ["OPENAI_API_KEY"],
        "seller_model": os.environ["SELLER_MODEL"],
        "api_base": os.environ["OPENAI_API_BASE"],
        "dataset_path": os.getenv("DATASET_PATH", DEFAULT_DATASET_PATH),
        "hf_dataset_repo": os.getenv("HF_DATASET_REPO", DEFAULT_HF_DATASET_REPO),
        "hf_dataset_split": os.getenv("HF_DATASET_SPLIT", DEFAULT_HF_DATASET_SPLIT),
        "hf_token": os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
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


def _normalize_dataset_rows(rows: list[dict]) -> Dataset:
    """Normalize rows into the prompt/info shape expected by the environment."""
    normalized = []
    for row in rows:
        if "prompt" in row and "info" in row:
            normalized.append({"prompt": row["prompt"], "info": row["info"]})
        elif "buyer_prompt" in row:
            normalized.append(
                {
                    "prompt": [{"role": "system", "content": row["buyer_prompt"]}],
                    "info": row,
                }
            )
        else:
            raise ValueError("Dataset rows must contain either (prompt, info) or buyer_prompt.")
    return Dataset.from_list(normalized)


def _load_env_dataset(
    dataset_path: str,
    hf_dataset_repo: str,
    hf_dataset_split: str = DEFAULT_HF_DATASET_SPLIT,
    hf_token: str | None = None,
) -> tuple[Dataset, str]:
    """
    Load dataset with priority:
    1) Hugging Face dataset repo/split
    2) Local JSON fallback (dataset_path)
    """
    hf_error = None
    if hf_dataset_repo:
        try:
            ds = load_dataset(
                hf_dataset_repo,
                split=hf_dataset_split,
                token=hf_token,
            )
            rows = list(ds)
            return _normalize_dataset_rows(rows), f"hf://{hf_dataset_repo}[{hf_dataset_split}]"
        except Exception as e:
            hf_error = e

    try:
        with open(dataset_path) as f:
            episodes = json.load(f)
        return _normalize_dataset_rows(episodes), dataset_path
    except Exception as local_error:
        msg = (
            "Failed to load dataset from Hugging Face and local fallback.\n"
            f"HF repo: {hf_dataset_repo} split={hf_dataset_split}\n"
            f"HF error: {type(hf_error).__name__}: {hf_error}\n"
            f"Local path: {dataset_path}\n"
            f"Local error: {type(local_error).__name__}: {local_error}"
        )
        raise RuntimeError(msg) from local_error
