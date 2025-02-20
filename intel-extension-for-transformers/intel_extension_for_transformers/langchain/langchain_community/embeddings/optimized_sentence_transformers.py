
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
    

    def _load_auto_model(
            self,
            model_name_or_path: str,
            token: Optional[Union[bool, str]],
            cache_folder: Optional[str],
            revision: Optional[str] = None,
            trust_remote_code: bool = False,
    ):
        """Creates a simple Transformer + Mean Pooling model and returns the modules."""
        logger.warning(
            "No sentence-transformers model found with name {}." \
            "Creating a new one with MEAN pooling.".format(model_name_or_path)
        )
        transformer_model = OptimizedTransformer(
            model_name_or_path,
            cache_dir=cache_folder,
            model_args={"token": token, "trust_remote_code": trust_remote_code, "revision": revision},
            tokenizer_args={"token": token, "trust_remote_code": trust_remote_code, "reivision": revision},
        )

        if isinstance(transformer_model.auto_model, torch.jit.ScriptModule):
            self._jit_model = True
        pooling_model = sentence_transformers.models.Pooling(
            transformer_model.get_word_embedding_dimension(), 'mean'
        )

        return [transformer_model, pooling_model]
    

    