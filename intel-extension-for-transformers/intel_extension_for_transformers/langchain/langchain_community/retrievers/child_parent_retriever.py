"""The wrapper for Child-Parent retriever based on langchain."""
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.pydantic_v1. import Field
from langchain_core.documents import Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun


from enum import Enum
from typing import List



class SearchType(str, Enum):
    """Enumerator of the types of search to perform."""

    similarity = "similarity"
    """Similarity search."""
    mmr = "mmr"
    """Maximal Marginal Relevance reranking of similarity search."""



class ChildParentRetriever(BaseRetriever):
    """Retrieve from a set of multiple embeddings for the same doc."""
    vectorstore: VectorStore
    parentstore: VectorStore
    id_key: str = "doc_id"
    search_kwargs: dict = Field(default_factory=dict)
    """Keyword args to pass to the search func."""
    search_type: SearchType = SearchType.similarity
    """Type of search to perform (similarty / mmr)"""


    def _get_relevant_documents(
            self,
            query: str, 
            *,
            run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Get docs relevant to a query.


        Args
        ----------------
            query: String to find relevant docs for
            run_manager: The callbacks handler to use


        Returns
        -----------------
            The concatation of the retrieved docs and the link
        """
        ids = []
        results = []

        if self.search_type == SearchType.mmr:
            sub_docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            sub_docs = self.vectorstore.similarity_search(query, **self.search_kwargs)
        

        for d in sub_docs:
            if d.metadata["identify_id"] not in ids:
                ids.append(d.metadata['identify_id'])
        

        retrieved_documents = self.parentstore.get(ids)
        for i in range(len(retrieved_documents['ids'])):
            metadata = retrieved_documents['metadatas'][i]
            context = retrieved_documents['documents'][i]
            instance = Document(page_content=context, metadata=metadata)
            results.append(instance)
        
        return results