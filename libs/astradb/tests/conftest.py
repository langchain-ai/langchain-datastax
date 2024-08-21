"""General-purpose testing tools, such as:
- Ad-hoc embedding classes
"""

from __future__ import annotations

import json

from langchain_core.embeddings import Embeddings


class SomeEmbeddings(Embeddings):
    """Turn a sentence into an embedding vector in some way.
    Not important how. It is deterministic is all that counts.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(txt) for txt in texts]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        unnormed0 = [ord(c) for c in text[: self.dimension]]
        unnormed = (unnormed0 + [1] + [0] * (self.dimension - 1 - len(unnormed0)))[
            : self.dimension
        ]
        norm = sum(x * x for x in unnormed) ** 0.5
        return [x / norm for x in unnormed]

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


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
            assert len(vals) == self.dimension
            return vals
        except Exception:
            print(f'[ParserEmbeddings] Returning a moot vector for "{text}"')
            return [0.0] * self.dimension

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)
