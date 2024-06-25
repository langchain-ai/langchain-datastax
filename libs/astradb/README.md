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

### LLM Cache

```python
from langchain_astradb import AstraDBCache

cache = AstraDBCache(
    api_endpoint="https://...",
    token="AstraCS:...",
)
```

### Semantic LLM Cache

```python
from langchain_astradb import AstraDBSemanticCache

cache = AstraDBSemanticCache(
    embedding=my_embedding,
    api_endpoint="https://...",
    token="AstraCS:...",
)
```

### Document loader

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

## Warnings about indexing

When creating an Astra DB object in LangChain, such as an `AstraDBVectorStore`, you may see a warning similar to the following:

> Astra DB collection '...' is detected as having indexing turned on for all fields (either created manually or by older versions of this plugin). This implies stricter limitations on the amount of text each string in a document can store. Consider reindexing anew on a fresh collection to be able to store longer texts.

The reason for the warning is that the requested collection already exists on the database, and it is configured to [index all of its fields for search](https://docs.datastax.com/en/astra-db-serverless/api-reference/collections.html#the-indexing-option), possibly implicitly, by default. When the LangChain object tries to create it, it attempts to enforce, instead, an indexing policy tailored to the prospected usage. For example, the LangChain vector store will index the metadata but leave the textual content out: this is both to enable storing very long texts and to avoid indexing fields that will never be used in filtering a search (indexing those would also have a slight performance cost for writes).

Typically there are two reasons why you may encounter the warning:

1. you have created a collection by other means than letting the `AstraDBVectorStore` do it for you: for example, through the Astra UI, or using AstraPy's `create_collection` method of class `Database` directly;
2. you have created the collection with a version of the Astra DB plugin that is not up-to-date (i.e. prior to the `langchain-astradb` partner package).

Keep in mind that this is a warning and your application will continue running just fine, as long as you don't store very long texts.
Should you need to add to a vector store, for example, a `Document` whose `page_content` exceeds ~8K in length, you will receive an indexing error from the database.

### Remediation

You have several options:

- you can ignore the warning because you know your application will never need to store very long textual contents;
- you can ignore the warning and explicitly instruct the plugin _not to_ create the collection, assuming it exists already (which suppresses the warning): `store = AstraDBVectorStore(..., setup_mode=langchain_astradb.utils.astradb.SetupMode.OFF)`. In this case the collection will be used as-is, no (indexing) questions asked;
- if you can afford populating the collection anew, you can drop it and re-run the LangChain application: the collection will be created with the optimized indexing settings. **This is the recommended option, when possible**.
