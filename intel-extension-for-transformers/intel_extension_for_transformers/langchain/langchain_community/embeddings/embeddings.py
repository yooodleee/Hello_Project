import logging
import importlib.util
from typing import Any, Dict, List, Optional


from .optimized_instructor_embedding import OptimizedInstructor
from .optimized_sentence_transformers import OptimizedSentenceTransformer
from intel_extension_for_transformers.transformers.utils.utility import LazyImport



langchain_core = LazyImport("langchain_core")

DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_INSTRUCT_MODEL = "hkunlp/instructor-large"
DEFAULT_BGE_MODEL = "BAAI/bge-large-en"
DEFAULT_EMBED_INSTRUCTION = "Represent the document for retrieval: "
DEFAULT_QUERY_INSTRUCTION = (
    "Represent the question for retrieving supporting documents: "
)
DEFAULT_QUERY_BGE_INSTRUCTION_EN = (
    "Represent this question for searching relevant passages: "
)
DEFAULT_QUERY_BGE_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文章："


logger = logging.getLogger(__name__)



