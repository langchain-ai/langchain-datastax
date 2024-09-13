import json
import logging
import os

from langchain_astradb.graph_vectorstores import AstraDBGraphVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.graph_vectorstores import Link
from langchain_core.graph_vectorstores.links import add_links


ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_DB_API_ENDPOINT = os.environ["ASTRA_DB_API_ENDPOINT"]
ASTRA_DB_KEYSPACE = os.environ["ASTRA_DB_KEYSPACE"]

# logging.basicConfig(level=5)


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
            print(f'[ParserEmbeddings] Returning a moot vector for "{text}"')
            return [0.0] * self.dimension
        else:
            assert len(vals) == self.dimension
            return vals

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


if __name__ == "__main__":

    # init & reset
    embeddings = ParserEmbeddings(dimension=2)
    graph_vectorstore = AstraDBGraphVectorStore(
        collection_name="graph_vs_test0",
        embedding=embeddings,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        namespace=ASTRA_DB_KEYSPACE,
        metric="euclidean",
    )
    graph_vectorstore.vectorstore.clear()

    # populate
    docs_a = [
        Document(page_content="[-1, 9]", metadata={"label": "AL"}),
        Document(page_content="[0, 10]", metadata={"label": "A0"}),
        Document(page_content="[1, 9]", metadata={"label": "AR"}),
    ]
    docs_b = [
        Document(page_content="[9, 1]", metadata={"label": "BL"}),
        Document(page_content="[10, 0]", metadata={"label": "B0"}),
        Document(page_content="[9, -1]", metadata={"label": "BR"}),
    ]
    docs_f = [
        Document(page_content="[1, -9]", metadata={"label": "BL"}),
        Document(page_content="[0, -10]", metadata={"label": "B0"}),
        Document(page_content="[-1, -9]", metadata={"label": "BR"}),
    ]
    docs_t = [
        Document(page_content="[-9, -1]", metadata={"label": "TL"}),
        Document(page_content="[-10, 0]", metadata={"label": "T0"}),
        Document(page_content="[-9, 1]", metadata={"label": "TR"}),
    ]
    for doc_a, suffix in zip(docs_a, ["l","0","r"]):
        add_links(doc_a, Link.bidir(kind="ab_example", tag=f"tag_{suffix}"))
        add_links(doc_a, Link.outgoing(kind="at_example", tag=f"tag_{suffix}"))
        add_links(doc_a, Link.incoming(kind="af_example", tag=f"tag_{suffix}"))
    for doc_b, suffix in zip(docs_b, ["l","0","r"]):
        add_links(doc_b, Link.bidir(kind="ab_example", tag=f"tag_{suffix}"))
    for doc_t, suffix in zip(docs_t, ["l","0","r"]):
        add_links(doc_t, Link.incoming(kind="at_example", tag=f"tag_{suffix}"))
    for doc_f, suffix in zip(docs_f, ["l","0","r"]):
        add_links(doc_f, Link.outgoing(kind="af_example", tag=f"tag_{suffix}"))

    """
    Space of the entries (under Euclidean similarity):

                      A0    (*)
                    AL   AR
        <...           |          <...>
                       |
                       |   :
                       |   :
       TR              |              BL
    T0   --------------x--------------   B0
       TL              |              BR
                       |   :
                       |   :
                       |   v
                       |
                    FL   FR
                      F0

    the query point is at (*).
    the A are bidirectionally with B
    the A are outgoing to T
    the A are incoming from F
    The links are like: L with L, 0 with 0 and R with R.
    """

    ad_response = graph_vectorstore.add_documents(
        docs_a + docs_b + docs_f + docs_t
    )
    print("\nad_response", ad_response)

    # regular search
    ss_response = graph_vectorstore.similarity_search(query="[2, 10]", k=2)
    print(f"\nss_response: {', '.join(doc.metadata['label'] for doc in ss_response)}")

    # traversal search
    ts_response = graph_vectorstore.traversal_search(query="[2, 10]", k=2, depth=2)
    print(f"\nts_response: {', '.join(doc.metadata['label'] for doc in ts_response)}")

    # mmr traversal search
    mt_response = graph_vectorstore.mmr_traversal_search(
        query="[2, 10]",
        k=2,
        depth=2,
        fetch_k=1,
        adjacent_k=2,
        lambda_mult=0.1,
    )
    print(f"\nmt_response: {', '.join(doc.metadata['label'] for doc in mt_response)}")
