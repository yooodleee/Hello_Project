from transformers import BarkModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BarkModel.from_pretrained(
    "suno/bark-small",
    torch_dtype=torch.float16,
).to(device)