
import os
import json
import logging
import torch
from typing import Union, Optional
from collections import OrderedDict

from intel_extension_for_transformers.transformers import OptimizedModel
from intel_extension_for_transformers.transformers.utils.utility import LazyImport
from transformers import T5Config, MT5Config


sentence_transformers = LazyImport("sentence_transformers")

WEIGHTS_NAME = "pytorch_model.bin"

logger = logging.getLogger(__name__)



class OptimizedTransformer(sentence_transformers.models.Transformer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    def _load_model(
            self,
            model_name_or_path,
            config,
            cache_dir,
            **model_args,
    ):
        """Loads the transformer model."""
        self.auto_model = OptimizedModel.from_pretrained(
            model_name_or_path, 
            config=config,
            cache_dir=cache_dir,
            **model_args,
        )

        if isinstance(self.auto_model, torch.jit.ScriptModule):
            setattr(self.auto_model, "config", config)
    

    