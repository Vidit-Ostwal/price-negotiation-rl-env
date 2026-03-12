import json

from generators.llm import HFLLMGenerator


class _DummyHFClient:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.calls = []
        self.chat = self._Chat(self)

    class _Chat:
        def __init__(self, outer) -> None:
            self.completions = outer._Completions(outer)

    class _Completions:
        def __init__(self, outer) -> None:
            self.outer = outer

        def create(self, model: str, messages: list[dict], **kwargs):
            self.outer.calls.append(
                {"model": model, "messages": messages, "kwargs": kwargs}
            )

            class _Message:
                def __init__(self, content: str) -> None:
                    self.content = content

            class _Choice:
                def __init__(self, content: str) -> None:
                    self.message = _Message(content)

            class _Response:
                def __init__(self, content: str) -> None:
                    self.choices = [_Choice(content)]

            return _Response(self.outer._response_text)


def test_hf_llm_generator_parses_and_validates(monkeypatch):
    """
    Smoke test HFLLMGenerator: ensure it can parse a JSON completion and
    return a validated product without hitting the real HF API.
    """

    product = {
        "name": "Vintage Test Camera",
        "description": "Well-kept 35mm film camera with minor cosmetic wear.",
        "market_price": 500,
        "haggle_norm": "medium",
        "typical_discount_pct": 20,
    }
    response_text = json.dumps(product)

    dummy_client = _DummyHFClient(response_text=response_text)

    # Patch the underlying InferenceClient used by HFLLMGenerator.
    import generators.llm as llm_mod

    monkeypatch.setattr(llm_mod, "InferenceClient", lambda *args, **kwargs: dummy_client)

    gen = HFLLMGenerator(
        model="dummy/model",
        token="dummy-token",
        api_base=None,
    )

    product_out = gen.generate("electronics")

    assert product_out["name"] == product["name"]
    assert dummy_client.calls, "HF client was not called"
