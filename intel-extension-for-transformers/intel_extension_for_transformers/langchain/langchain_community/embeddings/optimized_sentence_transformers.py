
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
    

    def forward(self, features):
        """Returns token_embeddings, cls_token."""
        trans_features = {
            "input_ids": features["input_ids"],
            "attention_mask": features["attention_mask"],
        }

        if "token_type_ids" in features:
            trans_features["token_type_ids"] = features["token_type_ids"]
        
        if isinstance(self.auto_model, torch.jit.ScriptModule):
            output_states = self.auto_model(**trans_features)
            if isinstance(output_states, dict):
                output_states = tuple(output_states.values())
            
            output_tokens = output_states[0]
        else:
            output_states = self.auto_model(**trans_features, return_dict=False)
            output_tokens = output_states[0]
        
        features.update(
            {
                "token_embeddings": output_tokens,
                "attention_mask": features["attention_mask"],
            }
        )

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3:
                all_layer_idx = 1
            
            hidden_states = output_states[all_layer_idx]
            features.update({"all_layer_embeddings": hidden_states})
        
        return features



class OptimizedSentenceTransformer(sentence_transformers.SentenceTransformer):

    def __init__(self, *args, **kwargs):
        self._jit_model = False
        super().__init__(*args, **kwargs)
    

    