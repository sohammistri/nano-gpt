import yaml
import torch
from typing import Any, Callable
import inspect

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    # Handle device separately
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    return config

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
