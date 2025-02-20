
from __future__ import annotations


from typing import TYPE_CHECKING, Dict, Optional, Sequence
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Extra, root_validator


from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from FlagEmbedding import FlagReranker



class BgeReranker(BaseDocumentCompressor):
    top_n: int = 3  # num of documents to return.
    model: FlagReranker
    """CrossEncoder instance to use for reranking."""

    def bge_reranker(self, query, docs):
        model_inputs = [[query, doc] for doc in docs]
        scores = self.model.compute_score(model_inputs)
        
        if len(docs) == 1:
            return [(0, scores)]
        results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        return results[:self.top_n]


    class Config:
        """Configuration for this pydantic obj."""

        extra = Extra.forbid
        arbitrary_types_allowed = True
    

    def compress_documents(
            self,
            documents: Sequence[Document],
            query: str,
            callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress docs using BAAI/bge-reranker models.
        

        Args
        -----------------
            documents: A sequence of docs to comrpress.
            query: The query to use for comrpressing the docs.
            callbacks: Callbacks to run during compression process.


        Returns
        -----------------
            A seq of compressed docs.
        """
        if len(documents) == 0: # to avoid empty api call
            return []
        
        doc_list = list(documents)
        _docs = [d.page_content for d in doc_list]
        results = self.bge_reranker(query, _docs)
        final_results = []

        for sample in results:
            doc = doc_list[sample[0]]
            doc.metadata["relevance_score"] = sample[1]
            final_results.append(doc)
        
        return final_results