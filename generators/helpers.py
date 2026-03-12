"""
Helper utilities for dataset generation:

- .env loading
- LLM and HF-LLM environment validation
- Hugging Face Hub push helpers
- Progress bar rendering
- Category planning for balanced/unbalanced sampling
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

from datasets import Dataset, concatenate_datasets, load_dataset

from generators.base import CATEGORIES


def _load_dotenv(dotenv_path: str = ".env") -> None:
    """Load KEY=VALUE pairs into env when keys are missing or empty."""
    env_file = Path(dotenv_path)
    if not env_file.exists():
        return

    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and not os.getenv(key):
            os.environ[key] = value


def _validate_llm_env() -> dict:
    """
    Validate environment variables used by LLM generation mode.

    Required:
    - OPENAI_API_KEY

    Optional (defaults applied when unset):
    - OPENAI_API_BASE (default: https://api.openai.com/v1)
    - GENERATOR_MODEL (default: gpt-4o-mini)
    """
    _load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: missing required env var OPENAI_API_KEY for --mode llm")
        print("Set it in your shell or .env before running.")
        sys.exit(1)

    api_base = os.getenv("OPENAI_API_BASE")
    model = os.getenv("GENERATOR_MODEL")

    if not api_base:
        print("Info: OPENAI_API_BASE not set, defaulting to https://api.openai.com/v1")
        api_base = "https://api.openai.com/v1"

    if not model:
        print("Info: GENERATOR_MODEL not set, defaulting to gpt-4o-mini")
        model = "gpt-4o-mini"

    return {
        "api_key": api_key,
        "api_base": api_base,
        "model": model,
    }


def _validate_hf_llm_env() -> dict:
    """
    Validate environment variables used by Hugging Face LLM generation mode.

    Required:
    - HF_LLM_MODEL (Hugging Face model id or endpoint id)
    - HF_TOKEN or HUGGINGFACE_HUB_TOKEN

    Optional:
    - HF_LLM_API_BASE (custom inference endpoint base URL)
    """
    _load_dotenv()

    model = os.getenv("HF_LLM_MODEL")
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    api_base = os.getenv("HF_LLM_API_BASE")

    missing = []
    if not model:
        missing.append("HF_LLM_MODEL")
    if not token:
        missing.append("HF_TOKEN / HUGGINGFACE_HUB_TOKEN")

    if missing:
        print("Error: missing required env vars for --mode hf-llm:")
        for m in missing:
            print(f"  - {m}")
        print("Set them in your shell or .env before running.")
        sys.exit(1)

    if not api_base:
        print("Info: HF_LLM_API_BASE not set; using default Hugging Face model endpoint.")

    return {
        "model": model,
        "token": token,
        "api_base": api_base,
    }


def _resolve_hf_push_env(repo_id_arg: Optional[str], token_arg: Optional[str]) -> dict:
    """
    Resolve and validate Hugging Face Hub settings for dataset push.

    Supported env vars:
    - HF_DATASET_REPO or HF_REPO_ID
    - HF_TOKEN or HUGGINGFACE_HUB_TOKEN
    """
    _load_dotenv()

    repo_id = repo_id_arg or os.getenv("HF_DATASET_REPO") or os.getenv("HF_REPO_ID")
    token = token_arg or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    if not repo_id:
        print("Error: missing Hugging Face dataset repo id.")
        print("Set --hf-repo-id or one of: HF_DATASET_REPO, HF_REPO_ID")
        sys.exit(1)

    if not token:
        print("Error: missing Hugging Face token.")
        print("Set --hf-token or one of: HF_TOKEN, HUGGINGFACE_HUB_TOKEN")
        sys.exit(1)

    return {
        "repo_id": repo_id,
        "token": token,
    }


def push_dataset_to_hf(
    episodes: list,
    repo_id: str,
    token: str,
    split: str = "train",
    private: bool = False,
    write_mode: str = "overwrite",
    commit_message: Optional[str] = None,
) -> None:
    """Push generated episodes to Hugging Face Hub as a datasets split."""
    if not episodes:
        print("Error: no episodes to push.")
        sys.exit(1)

    ds_new = Dataset.from_list(episodes)
    ds_to_push = ds_new

    if write_mode == "append":
        try:
            ds_existing = load_dataset(repo_id, split=split, token=token)
            ds_to_push = concatenate_datasets([ds_existing, ds_new])
            print(
                f"  HF append mode: existing={len(ds_existing)} + new={len(ds_new)} -> total={len(ds_to_push)}"
            )
        except Exception as e:
            print(
                "  HF append mode: could not load existing split "
                f"({type(e).__name__}); creating split with new rows."
            )

    msg = commit_message or (
        f"{'Append' if write_mode == 'append' else 'Overwrite'} negotiation dataset "
        f"(new_rows={len(ds_new)}, split={split})"
    )
    ds_to_push.push_to_hub(
        repo_id=repo_id,
        split=split,
        token=token,
        private=private,
        commit_message=msg,
    )


def _render_progress(done: int, total: int, start_time: float, mode_label: str) -> None:
    """Render progress UI for generation in a log-friendly way."""
    bar_width = 32
    inner_width = 78
    ratio = (done / total) if total > 0 else 1.0
    filled = int(bar_width * ratio)
    bar = "#" * filled + "-" * (bar_width - filled)
    elapsed = time.time() - start_time
    rate = (done / elapsed) if elapsed > 0 else 0.0
    eta = ((total - done) / rate) if rate > 0 else 0.0

    def _box_line(text: str) -> str:
        return "|" + text[:inner_width].ljust(inner_width) + "|"

    if done == 0 and not getattr(_render_progress, "_drawn", False):
        # Initial call: print a full boxed header.
        lines = [
            "+" + "-" * inner_width + "+",
            _box_line(f" Dataset Generation ({mode_label})"),
            _box_line(f" Progress: [{bar}] {done:>4}/{total:<4} ({ratio*100:5.1f}%)"),
            _box_line(
                f" Speed: {rate:7.2f} ep/s   Elapsed: {elapsed:7.1f}s   ETA: {eta:7.1f}s"
            ),
            "+" + "-" * inner_width + "+",
        ]
        sys.stdout.write("\n".join(lines) + "\n")
        _render_progress._drawn = True

    # Always append a fresh, standalone progress line.
    line = (
        f"[{bar}] {done:>4}/{total:<4} "
        f"({ratio*100:5.1f}%)  {rate:6.2f} ep/s  ETA {eta:7.1f}s"
    )
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _build_category_plan(n: int, balanced: bool) -> list[Optional[str]]:
    """Build per-episode category plan."""
    if not balanced:
        return [None for _ in range(n)]

    plan: list[Optional[str]] = []
    per_cat = n // len(CATEGORIES)
    remainder = n % len(CATEGORIES)
    for i, cat in enumerate(CATEGORIES):
        count = per_cat + (1 if i < remainder else 0)
        plan.extend([cat] * count)
    return plan

