from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import time
import os
import torch.nn.functional as F
import random

# Import tokenizer from transformers (already imported above, but repeated here)
from transformers import AutoTokenizer

# Load the model path from environment variable 'RAW_MODEL_PATH'; default is None if not set
model_path = os.getenv('RAW_MODEL_PATH', None)
print('model_path:', model_path)

# Load the pre-trained causal language model, automatically choosing the appropriate
# torch data type and device distribution
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",    # Automatically select torch dtype
    device_map="auto"      # Automatically place model onto available device(s)
)

# Load the tokenizer corresponding to the model
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Define a set of custom special tokens for the application
sp_token_str = ["<|branch|>", "<|/branch|>", "<|title|>", "<|/title|>", "<|para|>", "<|/para|>"]
special_tokens_dict = {
    "additional_special_tokens": sp_token_str
}

# Add the special tokens to the tokenizer's vocabulary
tokenizer.add_special_tokens(special_tokens_dict)

# Resize the model's embedding layer to match 
# the tokenizer's vocabulary size after adding special tokens
model.resize_token_embeddings(len(tokenizer))

# Save the updated tokenizer and model to a specified directory for future use
tokenizer.save_pretrained("para_model/Qwen2.5-7B-Instruct-Async")
model.save_pretrained("para_model/Qwen2.5-7B-Instruct-Async")

# Print environment variable export statements for specific special tokens,
# showing their integer token IDs (useful for downstream processing)
print(f'export PARA_ST_TOKEN={tokenizer.encode("<|para|>")[-1]}')
print(f'export PARA_ED_TOKEN={tokenizer.encode("<|/para|>")[-1]}')
