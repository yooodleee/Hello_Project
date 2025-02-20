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



