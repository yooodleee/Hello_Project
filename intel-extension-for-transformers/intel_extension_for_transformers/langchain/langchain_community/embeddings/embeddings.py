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



class HuggingFaceEmbeddings(
    langchain_core.pydantic_v1.BaseModel, langchain_core.embeddings.Embeddings
):
    """
    HuggingFace sentence_transformers embedding models.
    
    To use, should have the ``sentence_transformers`` python package installed.
    
    Example:
        .. code-block: python
        
        from intel_extension_for_transformers.lanchain_community.embeddings import HuggingFaceEmbeddings
        
        model_name = "sentece-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    """

    client: Any # : : meta private
    model_name: str = DEFAULT_MODEL_NAME
    """Model name to use."""
    cache_folder: Optional[str] = None
    """Path to store models.
    
    Can be also set by SENTENCE_TRANSFORMERS_HOME env variable.
    """
    model_kwargs: Dict[str, Any] = langchain_core.pydandict_v1.Field(default_factory=dict)
    """Keyword args to pass to the model."""
    encode_kwargs: Dict[str, Any] = langchain_core.pydantic_v1.Field(default_factory=dict)
    """Keyword args to pass when calling the `encode` method of model."""
    multi_process: bool = False
    """Run encode() on multiple GPUs."""


    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)

        
        # Check sentence_transformers python package
        if importlib.util.find_spec("sentence_transformers") is None:   # pragma: no cover
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install -U sentence-transformers`."
            )
        
        self.client = OptimizedSentenceTransformer(
            self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
        )
    

    class config:
        """Configuration for this pydantic obj."""

        extra = langchain_core.pydantic_v1.Extra.forbid
    
    