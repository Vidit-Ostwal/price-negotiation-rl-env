import json
import logging
import re
from typing import Optional

from huggingface_hub import InferenceClient
from litellm import completion

from generators.base import ProductGenerator, validate_product

logger = logging.getLogger(__name__)

LLM_PROMPT = """Generate a unique second-hand marketplace listing for a {category} item.

Return ONLY a valid JSON object with exactly these fields:
{{
  "name": "specific item name",
  "description": "2-3 sentences on condition and notable features",
  "market_price": <realistic integer USD price>,
  "haggle_norm": "<low|medium|high>",
  "typical_discount_pct": <integer 5-40>
}}

Rules:
- Be specific: not "antique vase" but "1920s Sevres porcelain vase with cobalt glaze"
- description must mention specific negotiation-relevant condition details
- market_price must be realistic for the second-hand market
- No markdown, no explanation, return only the raw JSON object
"""


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text


class LLMGenerator(ProductGenerator):
    def __init__(self, model, api_key, api_base, max_retries=2):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.max_retries = max_retries

    def generate(self, category):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                product = self._call_llm(category)
                return validate_product(product)
            except Exception as e:
                last_error = e
                logger.warning(f"LLM generation attempt {attempt+1} failed: {e}")
        raise RuntimeError(
            f"LLM generation failed for category '{category}' after {self.max_retries} attempts"
        ) from last_error

    def _call_llm(self, category):
        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": LLM_PROMPT.format(category=category)}],
            api_base=self.api_base,
            api_key=self.api_key,
            max_tokens=300,
            temperature=1.0,
        )
        raw = response.choices[0].message.content.strip()
        raw = _strip_code_fences(raw)
        return json.loads(raw)


class HFLLMGenerator(ProductGenerator):
    """
    Product generator that calls a Hugging Face conversational model.

    This is analogous to LLMGenerator but uses huggingface_hub.InferenceClient
    chat completions instead of LiteLLM / OpenAI-style APIs.
    """

    def __init__(
        self,
        model: str,
        token: str,
        api_base: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.9,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.max_retries = max_retries

        # If api_base is provided, treat it as a custom endpoint URL.
        if api_base:
            self.client = InferenceClient(base_url=api_base, token=token)
        else:
            self.client = InferenceClient(model=model, token=token)

        self.generation_kwargs = {
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }

    def generate(self, category: str) -> dict:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                product = self._call_hf(category)
                return validate_product(product)
            except Exception as e:
                last_error = e
                logger.warning(
                    "HF LLM generation attempt %s failed for category %s: %s",
                    attempt + 1,
                    category,
                    e,
                )
        raise RuntimeError(
            f"HF LLM generation failed for category '{category}' after {self.max_retries} attempts"
        ) from last_error

    def _call_hf(self, category: str) -> dict:
        prompt = LLM_PROMPT.format(category=category)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **self.generation_kwargs,
        )
        raw = response.choices[0].message.content.strip()
        raw = _strip_code_fences(raw)

        # Some models occasionally omit the outer braces and start directly
        # with `"name": ...`. Try to recover by slicing the first JSON object
        # span between the first '{' and last '}' before giving up.
        text = raw.lstrip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and start < end:
            text = text[start : end + 1]
        else:
            # Best effort: if the text starts with a quote, wrap in braces.
            if text.startswith('"') or text.startswith("\n\""):
                text = "{\n" + text + "\n}"

        return json.loads(text)
