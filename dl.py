#!/usr/bin/env python

import typing as ty

import numpy as np

from .modules import install as install_package
from .arrays import multidimensional_shifting, ArrayLike

try:
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  import torch.optim as optim
  from torch.utils.data import DataLoader, Dataset
except ImportError:
  install_package('torch')
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  import torch.optim as optim
  from torch.utils.data import DataLoader, Dataset


try:
  from accelerate import Accelerator
  from accelerate import utils as accelerate_utils
except ImportError:
  install_package('accelerate')
  from accelerate import Accelerator
  from accelerate import utils as accelerate_utils


def set_random_seed(seed):
  """Set random seed for reproducibility."""
  accelerate_utils.set_seed(seed)


def send_to_tensor(ctx, ctx2idx: dict, device=torch.device) -> torch.Tensor:
  """Send data to tensor."""
  indices = [ctx2idx[w] for w in ctx]
  tensor = torch.tensor(indices, dtype=torch.long)
  return tensor.to(device)


def generate_random_train_test_data(
  skipgrams: ArrayLike,
  txt2dict: dict,
  test_size: float = 0.3,
  sample_size: int = 1,
  device=torch.device) -> ty.Tuple[ArrayLike, ArrayLike]:
  
  pivot = len(skipgrams) - int(len(skipgrams) * test_size)
  indices = multidimensional_shifting(len(skipgrams), sample_size, skipgrams).T[0]
  training_idx, test_idx = indices[:pivot], indices[pivot:]
  training, test = skipgrams[training_idx,:], skipgrams[test_idx,:]
  
  training = [(send_to_tensor(x, txt2dict, device=device), send_to_tensor(y, txt2dict, device=device))
              for x,y in training]
  test = [(send_to_tensor(x, txt2dict, device=device), send_to_tensor(y, txt2dict, device=device))
          for x,y in test]
  
  return training, test



if __name__ == "__main__":
  pass
