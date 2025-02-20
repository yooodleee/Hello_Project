
import os
import json
import torch
import logging
from collections import OrderedDict


from intel_extension_for_transformers.transformers import OptimizeModel
from intel_extension_for_transformers.transformers.utils.utility import LazyImport
from transformers import T5Config, MT5Config


from typing import Union, Optional
from .optimized_sentence_transformers import OptimizedTransformer



sentence_transformers = LazyImport("sentence_transformers")
InstructorEmbedding = LazyImport("InstructorEmbedding")

logger = logging.getLogger(__name__)



class OptimizedInstructorTransformer(InstructorEmbedding.INSTRUCTOR_Transformer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    