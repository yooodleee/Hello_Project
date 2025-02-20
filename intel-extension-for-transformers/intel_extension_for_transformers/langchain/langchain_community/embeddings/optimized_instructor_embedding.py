
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
    

    def _load_model(
            self,
            model_name_or_path,
            config,
            cache_dir,
            **model_args,
    ):
        """Loads the transformer model."""
        self.auto_model = OptimizedTransformer.from_pretrained(
            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
        )

        if isinstance(self.auto_model, torch.jit.ScriptModule):
            setattr(self.auto_model, "config", config)
    

    def forward(self, features):
        """Returns token embeddings, cls_token."""
        trans_features = {
            'input_ids': features['input_ids'],
            'attention_mask': features['attention_mask'],
        }

        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']
        
        context_masks = None
        if 'context_masks' in features:
            context_masks = features['context_masks']
        
        if isinstance(self.auto_model, torch.jit.ScriptModule):
            output_states = self.auto_model(**trans_features)
            if isinstance(output_states, dict):
                output_states = tuple(output_states.values())
            
            output_tokens = output_states[0]
        
        else:
            output_states = self.auto_model(**trans_features, return_dict=False)
            output_tokens = output_states[0]
        
        attention_mask = features['attention_mask']
        if context_masks is not None:
            assert len(context_masks) == len(attention_mask)
            n = len(attention_mask)
            for local_idx in range(n):
                assert torch.sum(attention_mask[local_idx]).item() >= context_masks[local_idx].item(), \
                    f'{attention_mask[local_idx]}, {context_masks[local_idx]}, ' \
                    f'{torch.sum(attention_mask[local_idx]).item()}, {context_masks[local_idx].item()}'
                
                attention_mask[local_idx][:context_masks[local_idx]] = 0
        
        features.update({'token_embeddings': output_tokens, 'attention_mask': attention_mask})

        if self.auto_model.config.output_hidden_states:
            ally_layers_idx = 2
            if len(output_states) < 3: # some models only output last_hidden_states and all_hidden_states
                ally_layers_idx = 1
            
            hidden_states = output_states[ally_layers_idx]
            features.update({'all_layer_embeddings': hidden_states})

        return features



class OptimizedInstructor(InstructorEmbedding.INSTRUCTOR):

    def __init__(self, *args, **kwargs):
        self._jit_model = False
        super().__init__(*args, **kwargs)
    

    def _load_auto_model(
            self,
            model_name_or_path,
            token: Optional[Union[bool, str]],
            cache_folder: Optional[str] = None,
            trust_remote_code: bool = False,
    ):
        """Creates a simple Transformer + Mean Pooling model and returns the modules."""
        logger.warning(
            "No sentence-transformers model found with name {}." \
            "Creating a new one with MEAN pooling.".format(model_name_or_path)
        )
        transformer_model = OptimizedTransformer(
            model_name_or_path,
            cache_folder=cache_folder,
            model_args={'token': token, 'trust_remote_code': trust_remote_code, 'revision': revision},
            tokenizer_args={'token', token, 'trust_remote_cache': trust_remote_code, 'revision': revision},
        )
        if isinstance(transformer_model.auto_model, torch.jit.ScriptModule):
            self._jit_model = True
        pooling_model = sentence_transformers.models.Pooling(
            transformer_model.get_word_embedding_dimension(), 'mean'
        )

        return [transformer_model, pooling_model]


    