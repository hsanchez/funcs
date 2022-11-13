#!/usr/bin/env python

import typing as ty

import numpy as np
from tqdm import tqdm

from .arrays import ArrayLike, multidimensional_shifting
from .common import take
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


# defining the Dataset class
class SkipgramData(Dataset):
  def __init__(self, skipgram_data):
    self.data = skipgram_data

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    return self.data[index]


class SkipgramModel(nn.Module):
  def __init__(self, embedding_size: int, act2idx: dict, bias: bool = False, device = None) -> None:
    super(SkipgramModel, self).__init__()
    acts_size = len(act2idx)
    self.embedding = nn.Embedding(acts_size, embedding_size)
    self.W = nn.Linear(embedding_size, embedding_size, bias=bias) 
    self.WT = nn.Linear(embedding_size, acts_size, bias=bias)
    self.vocab = act2idx
    self.device = device
  
  def forward(self, X):
    embeddings = self.embedding(X)
    hidden_layer = nn.functional.relu(self.W(embeddings)) 
    output_layer = self.WT(hidden_layer)
    return output_layer
  
  def get_embedding(self, activity: str) -> torch.Tensor:
    if activity not in self.vocab:
      raise ValueError(f"Activity {activity} not in index")
    if self.device is not None:
      act_vec = torch.tensor([self.act2idx[activity]]).to(self.device)
    else:
      act_vec = torch.tensor([self.act2idx[activity]])
    return self.embedding(act_vec).view(1,-1)


def set_random_seed(seed):
  """Set random seed for reproducibility."""
  accelerate_utils.set_seed(seed)


def get_dataloaders_per_time_slice(
  train_test_by_time_slice: ty.List[tuple], 
  time_slices: ArrayLike, 
  batch_size: int = 10) -> ty.List[tuple]:
  if isinstance(time_slices, np.ndarray):
    time_slices = time_slices.tolist()

  train_test_dataloaders_by_time_slice = []
  for week in time_slices:
    train_w, test_w = train_test_by_time_slice[time_slices.index(week)]

    train_dataset = SkipgramData(train_w)
    test_dataset  = SkipgramData(test_w)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    train_test_dataloaders_by_time_slice.append((train_data_loader, test_data_loader))
  return train_test_dataloaders_by_time_slice


def accelerate_model(
  train_test_dataloaders: ty.List[tuple],
  timeline_slices: ArrayLike, 
  act2idx: dict, 
  embedding_size: int = 120, 
  learn_rate: float = 0.001, 
  accelerator: Accelerator = Accelerator()) -> ty.Tuple[Accelerator, ty.List[tuple]]:
  
  if isinstance(timeline_slices, np.ndarray):
    timeline_slices = timeline_slices.tolist()
  
  accelerations = []
  for t in timeline_slices:
    train_data_loader, test_data_loader = train_test_dataloaders[timeline_slices.index(t)]
    model = SkipgramModel(embedding_size, act2idx, accelerator.device)
    model, optimizer, train_data, test_data = accelerator.prepare(
      model,
      optim.Adam(model.parameters(), lr=learn_rate),
      train_data_loader,
      test_data_loader)
    accelerations += [(model, optimizer, train_data, test_data)]
  
  return accelerations, accelerator


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
  
  if isinstance(time_slices, np.ndarray):
    time_slices = time_slices.tolist()
  
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


def take_n_from_data_loader(n: int, t: int, train_test_dataloaders_by_time_slice: ty.List[tuple]) -> list:
  return take(n, train_test_dataloaders_by_time_slice[t][0])


if __name__ == "__main__":
  pass
