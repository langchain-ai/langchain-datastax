from langchain_astradb.graph_vectorstores import AstraDBGraphVectorStore
from langchain_community.graph_vectorstores.links import Link
from langchain_core.documents import Document

from typing import Callable

class GraphUpgradeUtility():
    gvs_store: AstraDBGraphVectorStore


    def upgrade_documents(
            self,
            link_function: Callable[[Document], set[Link]],
            batch_size = 10,
    ) -> int:
        filter = {"upgraded": {"$exists": False}}
        chunks = self.gvs_store.metadata_search(filter=filter, n=batch_size)
        if len(chunks) == 0:
            return 0

        id_to_md_map: dict[str, dict] = {}

        for chunk in chunks:
            links = link_function(chunk)
            new_metadata = self._convert_links_to_metadata(links=links)
            new_metadata["upgraded"] = True
            id_to_md_map[chunk.id] = new_metadata

        return self.gvs_store.update_metadata(id_to_metadata=id_to_md_map)



