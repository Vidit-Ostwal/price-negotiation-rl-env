"""
generate_dataset.py
====================
Single entry point for dataset generation.

Usage:
    # Template mode (default) -- fast, no API calls
    python generators/generate_dataset.py --n 100 --output dataset.json

    # LLM mode -- unique products every episode, requires API
    python generators/generate_dataset.py --n 100 --mode llm --output dataset.json
"""

import json
import random
import time
import argparse
from typing import Optional

from generators.base import CATEGORIES
from generators.template import TemplateGenerator
from generators.episode import generate_episode
from generators.helpers import (
    _build_category_plan,
    _load_dotenv,
    _render_progress,
    _resolve_hf_push_env,
    _validate_hf_llm_env,
    _validate_llm_env,
    push_dataset_to_hf,
)

# ---------------------------------------------------------------------------
# DATASET GENERATOR  (unchanged)
# ---------------------------------------------------------------------------

def generate_dataset(generator, n: int, balanced: bool = True, show_tui: bool = True) -> list:
    episodes = []
    generated = 0
    start_time = time.time()
    mode_label = "balanced" if balanced else "unbalanced"

    if show_tui:
        _render_progress._drawn = False
        _render_progress(done=0, total=n, start_time=start_time, mode_label=mode_label)

    def _append_episode(cat=None):
        nonlocal generated
        episodes.append(generate_episode(generator, cat))
        generated += 1
        if show_tui:
            _render_progress(done=generated, total=n, start_time=start_time, mode_label=mode_label)

    if balanced:
        per_cat   = n // len(CATEGORIES)
        remainder = n % len(CATEGORIES)
        for i, cat in enumerate(CATEGORIES):
            count = per_cat + (1 if i < remainder else 0)
            for _ in range(count):
                _append_episode(cat)
    else:
        for _ in range(n):
            _append_episode()

    random.shuffle(episodes)

    difficulties = [e["valuations"]["difficulty"] for e in episodes]
    print(f"\n  Generated {len(episodes)} episodes")
    for d in ["easy", "medium", "hard", "no_deal"]:
        c = difficulties.count(d)
        print(f"  {d:10s}: {c:4d}  ({100*c/len(episodes):.1f}%)")

    return episodes


def generate_dataset_with_checkpoints(
    generator,
    n: int,
    balanced: bool,
    checkpoint_size: int,
    checkpoint_callback,
    show_tui: bool = True,
) -> list:
    """
    Generate dataset and invoke checkpoint_callback every checkpoint_size episodes.
    callback signature: callback(checkpoint_rows: list[dict], generated_count: int, total_count: int)
    """
    episodes: list[dict] = []
    chunk: list[dict] = []
    generated = 0
    start_time = time.time()
    mode_label = "balanced" if balanced else "unbalanced"
    plan = _build_category_plan(n=n, balanced=balanced)

    if show_tui:
        _render_progress._drawn = False
        _render_progress(done=0, total=n, start_time=start_time, mode_label=mode_label)

    for cat in plan:
        ep = generate_episode(generator, cat)
        episodes.append(ep)
        chunk.append(ep)
        generated += 1

        if show_tui:
            _render_progress(done=generated, total=n, start_time=start_time, mode_label=mode_label)

        if len(chunk) >= checkpoint_size:
            checkpoint_callback(chunk, generated, n)
            chunk = []

    if chunk:
        checkpoint_callback(chunk, generated, n)

    random.shuffle(episodes)

    difficulties = [e["valuations"]["difficulty"] for e in episodes]
    print(f"\n  Generated {len(episodes)} episodes")
    for d in ["easy", "medium", "hard", "no_deal"]:
        c = difficulties.count(d)
        print(f"  {d:10s}: {c:4d}  ({100*c/len(episodes):.1f}%)")

    return episodes

# ---------------------------------------------------------------------------
# MAIN  -- single entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate negotiation dataset")
    parser.add_argument("--n",        type=int,   default=100,            help="Number of episodes")
    parser.add_argument("--output",   type=str,   default="dataset.json", help="Output file path")
    parser.add_argument(
        "--mode",
        type=str,
        default="template",
        choices=["template", "llm", "hf-llm"],
        help="Generator mode",
    )
    bal_group = parser.add_mutually_exclusive_group()
    bal_group.add_argument("--balanced",   dest="balanced", action="store_true",  default=True,  help="Balance across categories")
    bal_group.add_argument("--unbalanced", dest="balanced", action="store_false", help="Sample categories randomly")
    parser.add_argument("--seed",     type=int,   default=42,             help="Random seed")
    parser.add_argument("--push-to-hf",      action="store_true", help="Push generated dataset to Hugging Face Hub")
    parser.add_argument("--hf-repo-id",      type=str, default=None, help="HF dataset repo id (e.g. username/repo)")
    parser.add_argument("--hf-token",        type=str, default=None, help="HF token (or set HF_TOKEN in env)")
    parser.add_argument("--hf-split",        type=str, default="train", help="Dataset split name for Hub push")
    parser.add_argument("--hf-private",      action="store_true", help="Create/update private dataset repo")
    parser.add_argument(
        "--hf-write-mode",
        type=str,
        choices=["overwrite", "append"],
        default="append",
        help="How to write to HF split: overwrite existing split or append to it",
    )
    parser.add_argument(
        "--hf-push-every",
        type=int,
        default=100,
        help="When using --mode llm + --push-to-hf, append checkpoints every N episodes",
    )
    parser.add_argument("--hf-commit-message", type=str, default=None, help="Optional Hub commit message")
    args = parser.parse_args()

    random.seed(args.seed)
    _load_dotenv()

    # Select generator
    if args.mode == "llm":
        from generators.llm import LLMGenerator

        llm_env = _validate_llm_env()
        generator = LLMGenerator(
            model=llm_env["model"],
            api_key=llm_env["api_key"],
            api_base=llm_env["api_base"],
        )
        print(f"Mode: LLM ({llm_env['model']}) -- each product uniquely generated")
    elif args.mode == "hf-llm":
        from generators.llm import HFLLMGenerator

        hf_env = _validate_hf_llm_env()
        generator = HFLLMGenerator(
            model=hf_env["model"],
            token=hf_env["token"],
            api_base=hf_env["api_base"],
        )
        print(f"Mode: HF LLM ({hf_env['model']}) -- each product uniquely generated")
    else:
        generator = TemplateGenerator()
        print("Mode: template -- using hardcoded product bank")

    checkpoint_push = args.push_to_hf and args.mode in {"llm", "hf-llm"}
    hf_env = None
    if checkpoint_push:
        hf_env = _resolve_hf_push_env(args.hf_repo_id, args.hf_token)
        checkpoint_size = max(1, args.hf_push_every)
        print(f"Generating {args.n} episodes with checkpoint pushes every {checkpoint_size} episodes...")

        def _push_checkpoint(rows: list[dict], generated_count: int, total_count: int) -> None:
            start_idx = generated_count - len(rows) + 1
            end_idx = generated_count
            print(
                f"\n  HF checkpoint push ({start_idx}-{end_idx}/{total_count}) "
                f"to {hf_env['repo_id']}[{args.hf_split}]"
            )
            push_dataset_to_hf(
                episodes=rows,
                repo_id=hf_env["repo_id"],
                token=hf_env["token"],
                split=args.hf_split,
                private=args.hf_private,
                write_mode=args.hf_write_mode,
                commit_message=(
                    args.hf_commit_message
                    or (
                        f"{'Append' if args.hf_write_mode == 'append' else 'Overwrite'} "
                        f"checkpoint rows {start_idx}-{end_idx} (split={args.hf_split})"
                    )
                ),
            )

        dataset = generate_dataset_with_checkpoints(
            generator=generator,
            n=args.n,
            balanced=args.balanced,
            checkpoint_size=checkpoint_size,
            checkpoint_callback=_push_checkpoint,
        )
    else:
        print(f"Generating {args.n} episodes...")
        dataset = generate_dataset(generator, args.n, balanced=args.balanced)

    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n  Saved to {args.output}")

    if args.push_to_hf and not checkpoint_push:
        hf_env = _resolve_hf_push_env(args.hf_repo_id, args.hf_token)
        print(f"  Pushing to Hugging Face Hub: {hf_env['repo_id']} (split={args.hf_split})")
        push_dataset_to_hf(
            episodes=dataset,
            repo_id=hf_env["repo_id"],
            token=hf_env["token"],
            split=args.hf_split,
            private=args.hf_private,
            write_mode=args.hf_write_mode,
            commit_message=args.hf_commit_message,
        )
        print(f"  Push complete: https://huggingface.co/datasets/{hf_env['repo_id']}")

    ep = dataset[0]
    print(f"\n  Sample: {ep['product']['name']} ({ep['valuations']['difficulty']})")
    print(f"  Buyer ${ep['valuations']['buyer_true_value']} | Seller ${ep['valuations']['seller_reserve_price']} | ZOPA {ep['valuations']['zopa']}")
