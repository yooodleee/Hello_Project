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
    

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Compute doc embeddings using a HuggingFace transformer model.


        Args
        --------------------
            texts: The list of texts to embed.

        Returns
        ---------------
            List of embeddings, one for each text.
        """
        import sentence_transformers

        texts = list(map(lambda x: x.replace("\n", " "), texts))
        if self.multi_process:  # pragma: no cover
            pool = self.client.start_multi_process_pool()
            embeddings = self.client.encode_multi_process(texts, pool)
            sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
        
        else:
            embeddings = self.client.encode(texts, **self.encode_kwargs)
        
        return embeddings.tolist()
    

    def embed_query(self, text: str) -> List[float]:
        """
        Compute query embeddings using a HuggingFace transformer model.


        Args
        ----------------
            text: The text to embed.

        Returns
        ----------------
            Embeddings for the text.
        """
        
        return self.embed_documents([text])[0]



class HuggingFaceBgeEmbeddings(
    langchain_core.pydantic_v1.BaseModel, langchain_core.embeddings.Embeddings
):
    """
    HuggingFace BGE sentence_transformers embeddings models.
    
    To use, should have the ``sentence_transformers`` python package installed.


    Example:
        .. code-block: python

            from intel_extension_for_transformers.langchain_community.embeddings import HuggingFaceBgeEmbeddings

            model_name = "BAAI/bge-large-en"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': True}
            hf = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
    """

    client: Any # : : meta private:
    model_name: str = DEFAULT_BGE_MODEL
    """Model name to use."""
    cache_folder: Optional[str] = None
    """
    Path to store models.
    
    Can be also set by SENTENCE_TRANSFORMERS_HOME env variable.
    """
    model_kwargs: Dict[str, Any] = langchain_core.pydantic_v1.Field(default_factory=dict)
    """Keyword args to pass to the model."""
    encode_kwargs: Dict[str, Any] = langchain_core.pydantic_v1.Field(default_factory=dict)
    """Keyword args to pas when calling the `encode` method of the model."""
    query_instruction: str = DEFAULT_QUERY_BGE_INSTRUCTION_EN
    """Instruction to use for embedding query."""


    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)


        # Check sentence_transformers python package
        if importlib.util.find_spec("sentence_transformers") is None:
            raise ImportError(
                "Could not import sentece_transformers python package. "
                "Please install it with `pip install -U sentence-transformers`."
            )
        
        self.client = OptimizedSentenceTransformer(
            self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
        )
        if "-zh" in self.model_name:
            self.query_instruction = DEFAULT_QUERY_BGE_INSTRUCTION_ZH
    

    class config:
        """Configuration for this pydantic obj."""

        extra = langchain_core.pydantic_v1.Extra.forbid
    

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.
        
        
        Args
        --------------
            texts: The list of texts to embed.
            
        Returns
        ---------------
            List of embeddings, one for each text.
        """
        texts = [t.replace("\n", " ") for t in texts]
        embeddings = self.client.encode(texts, **self.encode_kwargs)

        return embeddings.tolist()
    

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggigFace transformer model.
        
        
        Args
        ---------------
            text: The text to embed.
            
            
        Returns
        ----------------
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        embedding = self.client.encode(
            self.query_instruction + text, **self.encode_kwargs
        )

        return embedding.tolist()



class HuggingFaceInstructEmbeddings(
    langchain_core.pydantic_v1.BaseModel, langchain_core.embeddings.Embeddings
):
    """
    Wrapper around sentence_transformers embedding models.

    To use, should have the ``sentence_transformers``
    and ``InstructorEmbedding`` python packages installed.


    Example:
        .. code-block: python

            from intel_extension_for_transformers.langchain_community.embeddings import HuggingFaceFaceInstructEmbeddings

            model_name = "hkunlp/instructor-large"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': True}
            hf = HuggingFaceInstructEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
    """

    client: Any
    model_name: str = DEFAULT_INSTRUCT_MODEL
    """Model name to use."""
    cache_folder: Optional[str] = None
    """Path to store models.
    
    Can be also set by SENTENCE_TRANSFORMERS_HOME env variable.
    """
    model_kwargs: Dict[str, Any] = langchain_core.pydantic_v1.Field(default_factory=dict)
    """Keyword args to pass to the model."""
    encode_kwargs: Dict[str, Any] = langchain_core.pydantic_v1.Field(default_factory=dict)
    """Keyword args to pass when calling the `encode` method of the model."""
    embed_instruction: str = DEFAULT_EMBED_INSTRUCTION
    """Instruction to use for embedding docs."""
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    """Instruction to use for embedding query."""


    def __init__(self, **kwargs: Any):
        """Init the sentence_transformer."""
        super().__init__(**kwargs)

        
        # Check sentece_transformers python package
        if importlib.util.find_spec("sentence_transformers") is None:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install -U sentece-transformers`."
            )
        
        # Check InstructorEmbedding python package
        if importlib.util.find_spec("InstructorEmbedding") is None:
            raise ImportError(
                "Could not import InstructorEmbedding python package. "
                "Please install it with `pip install -U InstructorEmbedding`."
            )
        
        self.client = OptimizedSentenceTransformer(
            self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
        )
    

    class Config:
        """Configuration for this pydantic obj."""

        extra = langchain_core.pydantic_v1.Extra.forbid

    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Compute doc embeddings using a HuggingFace instruct model.


        Args
        --------------
            texts: The list of texts to embed.


        Returns
        ---------------
            List of embeddings, one for each text.
        """
        instruct_pairs = [[self.embed_instruction, text] for text in texts]
        embeddings = self.client.encode(instruct_pairs, **self.encode_kwargs)

        return embeddings.tolist()
    

    def embed_query(self, text: str) -> List[float]:
        """
        Compute query embeddings using a HuggingFace instruct model.


        Args
        ----------------
            text: The text to embed.

        Returns
        ---------------
            Embeddings for the text.
        """
        instruction_pair = [self.query_instruction, text]
        embedding = self.client.encode([instruction_pair], **self.encode_kwargs)[0]

        return embedding.tolist()