import yaml
import torch
from typing import Any, Callable
import inspect
import pickle
import os

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def get_data(dataset, data_path, tokenizer, split_ratio=0.8, train_on_full=False):
    if dataset == "tiny_shakespeare":
        # get the raw tiny shakespeare dataset and split into train and val sets
        with open(data_path) as file:
            data = file.read()

        tokens = tokenizer.encode(data)

        split_idx = int(split_ratio * len(tokens))
        train_tokens = tokens[:split_idx]
        val_tokens = tokens[split_idx:]

        return train_tokens, val_tokens
    elif dataset == "1B_word_LM":
        # data_path is folder for this
        with open(os.path.join(data_path, 'train.pkl'), 'rb') as file:
            train_tokens = pickle.load(file)
        with open(os.path.join(data_path, 'val.pkl'), 'rb') as file:
            val_tokens = pickle.load(file)
        with open(os.path.join(data_path, 'test.pkl'), 'rb') as file:
            test_tokens = pickle.load(file)

        if train_on_full:
            train_tokens = train_tokens.extend(val_tokens)
            val_tokens = list(test_tokens)

        return train_tokens, val_tokens


def call_with_matching_args(func: Callable, data_dict: dict) -> Any:
    """
    Call a function or method with only the arguments it accepts from a dictionary.
    Works with both regular functions and class methods.
    """
    params = inspect.signature(func).parameters
    # Remove 'self' parameter if it's a method
    if 'self' in params:
        params = dict(list(params.items())[1:])
    filtered_args = {k: data_dict[k] for k in params if k in data_dict}
    return func(**filtered_args)

def get_batch(tokens, block_size, batch_size):
    batch = torch.randint(0, len(tokens)-block_size, (batch_size,)) # B dimension array of random indices
    Xb = torch.stack([torch.LongTensor(tokens[i:i+block_size]) for i in batch], dim=0) # Create (B, T) dimension array
    yb = torch.stack([torch.LongTensor(tokens[i+1:i+block_size+1]) for i in batch], dim=0) # Create (B, T) dimension array
    return Xb, yb

@torch.no_grad()
def compute_loss(tokens, block_size, batch_size, model, device):
    loss_values = []
    for _ in range(100):
        Xb, yb = get_batch(tokens, block_size, batch_size)
        Xb, yb = Xb.to(device), yb.to(device)

        _, loss = model(Xb, yb)
        loss_values.append(loss.item())

    mean_loss = torch.FloatTensor(loss_values).mean().item()
    return mean_loss
