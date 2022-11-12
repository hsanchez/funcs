#!/usr/bin/env python

import typing as ty

import numpy as np

from .arrays import ArrayLike, multidimensional_shifting
from .console import new_progress_display, quiet_stderr, stderr
from .modules import install as install_package

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


def get_train_test_data_per_period(
  sliced_skipgrams: ArrayLike, 
  time_slices: ArrayLike,
  txt2dict: dict,
  progress_bar: bool = False) -> ArrayLike:
  
  the_console = quiet_stderr
  if progress_bar:
    the_console = stderr
  
  train_test_data = []
  with new_progress_display(the_console) as progress:
    task = progress.add_task("Collecting data in time slices ...", total=len(sliced_skipgrams))
    for time_slice in time_slices:
      train_, test_ = generate_random_train_test_data(
        sliced_skipgrams[time_slices.index(time_slice)], txt2dict)
      train_test_data.append((train_, test_))
      progress.update(task, advance=1)
  
  return np.array(train_test_data)



def get_annotated_coordinates_from_model(activities: ArrayLike, trained_model: ty.Any, act2abbr: dict) -> tuple:
  """Get coordinates from tensor."""
  for activity in activities:
    embedding = trained_model.get_embedding(activity)
    if torch.cuda.is_available():
      x = embedding = embedding.detach().cpu().numpy()
      y = embedding = embedding.detach().cpu().numpy()
    else:
      x = embedding.detach().data.numpy()
      y = embedding.detach().data.numpy()
    yield act2abbr[activity], x, y


if __name__ == "__main__":
  pass
