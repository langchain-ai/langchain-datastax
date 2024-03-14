# langchain-astradb

This package contains the LangChain integrations for using DataStax Astra DB.

> DataStax [Astra DB](https://docs.datastax.com/en/astra/home/astra.html) is a serverless vector-capable database built on Apache Cassandra® and made conveniently available
> through an easy-to-use JSON API.

## Installation and Setup

Installation of this partner package:

```bash
pip install langchain-astradb
```

## Integrations overview

See the [LangChain docs page](https://python.langchain.com/docs/integrations/providers/astradb) and the [API reference](https://api.python.langchain.com/en/latest/astradb_api_reference.html) for more details.

### Vector Store

```python
from langchain_astradb import AstraDBVectorStore

my_store = AstraDBVectorStore(
  embedding=my_embedding,
  collection_name="my_store",
  api_endpoint="https://...",
  token="AstraCS:...",
)
```

### Chat message history

```python
from langchain_astradb import AstraDBChatMessageHistory

message_history = AstraDBChatMessageHistory(
    session_id="test-session",
    api_endpoint="https://...",
    token="AstraCS:...",
)
```

## LLM Cache

```python
from langchain_astradb import AstraDBCache

cache = AstraDBCache(
    api_endpoint="https://...",
    token="AstraCS:...",
)
```

## Semantic LLM Cache

```python
from langchain_astradb import AstraDBSemanticCache

cache = AstraDBSemanticCache(
    embedding=my_embedding,
    api_endpoint="https://...",
    token="AstraCS:...",
)
```

## Document loader

```python
from langchain_astradb import AstraDBLoader

loader = AstraDBLoader(
    collection_name="my_collection",
    api_endpoint="https://...",
    token="AstraCS:...",
)
```

### Store

```python
from langchain_astradb import AstraDBStore

store = AstraDBStore(
    collection_name="my_kv_store",
    api_endpoint="https://...",
    token="AstraCS:...",
)
```

### Byte Store

```python
from langchain_astradb import AstraDBByteStore

store = AstraDBByteStore(
    collection_name="my_kv_store",
    api_endpoint="https://...",
    token="AstraCS:...",
)
```
