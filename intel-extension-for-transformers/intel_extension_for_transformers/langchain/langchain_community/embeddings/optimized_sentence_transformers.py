
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
    

    def _load_sbert_model(
            self,
            model_name_or_path: str,
            token: Optional[Union[bool, str]],
            cache_folder: Optional[str],
            revision: Optional[str] = None,
            trust_remote_code: bool = False,
    ):
        """Loads a full sentence-transformers model."""

        # Check if the config_sentence_transformers.json file exists (exists since v2 of the framework)
        config_sentence_transformers_json_path = sentence_transformers.util.load_file_path(
            model_name_or_path,
            'config_sentence_transformers.json',
            token=token,
            cache_dir=cache_folder,
            revision=revision,
        )

        if config_sentence_transformers_json_path is not None:
            with open(config_sentence_transformers_json_path) as fIn:
                self._model_config = json.load(fIn)
            
            if '__version__' in self._model_config and \
                'sentence_transformers' in self._model_config['__version__'] and \
                self._model_config['__version__']['sentence_transformers'] > sentence_transformers.__version__:
                logger.warning(
                    "You try to use a model that was created with version {}, "\
                    "however, your version is {}. This might cause unexpected "\
                    "behavior or errors. In that case, try to update to the "\
                    "latest version.\n\n\n".format(
                        self._model_config['__version__']['sentence_transformers'],
                        sentence_transformers.__version__
                    )
                )
        

        # Check if a readme exists
        model_card_path = sentence_transformers.util.load_file_path(
            model_card_path, 'README.md', token=token, cache_folder=cache_folder, revision=revision
        )

        if model_card_path is not None:
            try:
                with open(model_card_path, encoding='utf8') as fIn:
                    self._model_card_text = fIn.read()
            except:
                pass
        

        # Load the modules of sentence transformer
        modules_json_path = sentence_transformers.util.load_file_path(
            model_card_path,
            'modules.json',
            token=token,
            cache_folder=cache_folder,
            revision=revision,
        )
        with open(modules_json_path) as fIn:
            modules_config = json.load(fIn)
        

        modules = OrderedDict()
        for module_config in modules_config:
            module_class = sentence_transformers.util.import_from_string(module_config['type'])
            # For Transformer, don't load the full dir, rely on 'transformers' instead
            # But, do load the config file first.
            if module_class == sentence_transformers.models.Transformer and module_config['path'] == "":
                kwargs = {}
                for config_name in [
                    'sentence_bert_config.json',
                    'sentence_roberta_config.json',
                    'sentence_distilbert_config.json',
                    'sentence_camembert_config.json',
                    'sentence_albert_config.son',
                    'sentence_xlm-roberta_config.json',
                    'sentence_xlnet_config.json',
                ]:
                    config_path = sentence_transformers.util.load_file_path(
                        model_name_or_path,
                        config_name,
                        token=token,
                        cache_folder=cache_folder,
                        revision=revision,
                    )

                    if config_path is not None:
                        with open(config_path) as fIn:
                            kwargs = json.load(fIn)
                        break
                

                hub_kwargs = {
                    "token": token,
                    "trust_remote_code": trust_remote_code,
                    "revision": revision,
                }

                if "model_args" in kwargs:
                    kwargs["model_args"].update(hub_kwargs)
                else:
                    kwargs["model_args"] = hub_kwargs
                
                if "tokenizer_args" in kwargs:
                    kwargs["tokenizer_args"].update(hub_kwargs)
                else:
                    kwargs["tokenizer_args"] = hub_kwargs
                
                module = OptimizedSentenceTransformer(
                    model_name_or_path, cahce_dir=cache_folder, **kwargs
                )

                if isinstance(module.auto_model, torch.jit.ScriptModule):
                    self._jit_model = True
            
            else:
                # Normalize doest not require any files to be loaded
                if module_class == sentence_transformers.models.Normalize:
                    module_path = None
                else:
                    module_path = sentence_transformers.util.load_dir_path(
                        model_name_or_path,
                        module_config["path"],
                        token=token,
                        cache_folder=cache_folder,
                        revision=revision,
                    )
                module = module_class.load(module_path)
            modules[module_config['name']] = module
        

        return modules
    

    def encode(
            self,
            sentences,
            device=None,
            *args,
            **kwargs,
    ):
        if self._jit_model and device is None:
            # set default device to 'cpu' for jit model, otherwise may fail when getting device
            return super().encode(sentences, device='cpu', *args, **kwargs)
        else:
            return super().encode(sentences, device=device, *args, **kwargs)