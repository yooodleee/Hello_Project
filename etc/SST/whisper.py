"""It is used to instantiate a Whisper model according to the specified args,
defining the model architecture. Instantiating a configuration with the defaults
will yield a similar configuration to that of the Whisper openai/whisper-tiny 
architecture.

Configuration objs inherit from PretrainedConfig and can be used to control the
model outputs.
"""
from transformers import WhisperConfig, WhisperModel, WhisperTokenizer


# Initializing a Whisper tiny style configuration
configuration = WhisperConfig()

# Initializing a model (with random weights) from the tiny style configuration
model = WhisperModel(configuration)

# Accessing the model configuration
configuration = model.config

# Instantiate the tokenizer and set the prefix token to Spanish
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="spanish")

# now switch the prefix token from Spanish to French
tokenizer.set_prefix_tokens(language="french")