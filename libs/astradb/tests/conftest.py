"""General-purpose testing tools, such as:
- Ad-hoc embedding classes
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import LLM
from typing_extensions import override

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForLLMRun


class ParserEmbeddings(Embeddings):
    """Parse input texts: if they are json for a List[float], fine.
    Otherwise, return all zeros and call it a day.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(txt) for txt in texts]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        try:
            vals = json.loads(text)
        except json.JSONDecodeError:
            return [0.0] * self.dimension
        else:
            assert len(vals) == self.dimension
            return vals

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


class IdentityLLM(LLM):
    num_calls: int = 0

    @property
    @override
    def _llm_type(self) -> str:
        return "fake"

    @override
    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        self.num_calls += 1
        if stop is not None:
            return f"STOP<{prompt.upper()}>"
        return prompt

    @property
    @override
    def _identifying_params(self) -> dict[str, Any]:
        return {}
