import re
import json
import logging
from litellm import completion
from generators.base import ProductGenerator, validate_product
from generators.template import TemplateGenerator

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


class LLMGenerator(ProductGenerator):
    def __init__(self, model, api_key, api_base, max_retries=2):
        self.model       = model
        self.api_key     = api_key
        self.api_base    = api_base
        self.max_retries = max_retries
        self._fallback   = TemplateGenerator()

    def generate(self, category):
        for attempt in range(self.max_retries):
            try:
                product = self._call_llm(category)
                return validate_product(product)
            except Exception as e:
                logger.warning(f"LLM generation attempt {attempt+1} failed: {e}")
        logger.warning(f"LLM generation failed for {category} -- using template fallback")
        return self._fallback.generate(category)

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
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw)
