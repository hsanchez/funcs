#!/usr/bin/env python

import typing as ty
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm import tqdm

from .arrays import ArrayLike, multidimensional_shifting, all_but_the_top
from .common import take
from .console import new_progress_display, new_quiet_console, stderr
from .modules import install as install_package
from .plots import plot_dynamic_activity_embeddings

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


try:
  from scipy.spatial import procrustes
except ImportError:
  install_package('scipy')
  from scipy.spatial import procrustes


try:
  from sklearn.metrics.pairwise import cosine_similarity as sim
  from sklearn.model_selection import train_test_split
except ImportError:
  install_package('scikit-learn')
  from sklearn.metrics.pairwise import cosine_similarity as sim
  from sklearn.model_selection import train_test_split


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


class AlignedW2V:
  def __init__(
    self, act2idx: dict, 
    idx2act: dict, 
    post_process_models: bool = True, 
    n_principal_components: int = 10) -> None:
    self.embeddings = []
    self.act2idx = act2idx
    self.idx2act = idx2act
    self.post_process_models = post_process_models
    self.n_principal_components = n_principal_components
  
  def fit(self, timeline_slices: ArrayLike, timeline_slice_models: ArrayLike):
    if isinstance(timeline_slices, np.ndarray):
      timeline_slices = timeline_slices.tolist()
    
    for time_slice in timeline_slices:
      time_slice_model = timeline_slice_models[timeline_slices.index(time_slice)]
      activity_embedding = np.array([get_tensor_data(time_slice_model.get_embedding(act))[0]
                                     for act in self.act2idx])
      if self.post_process_models and len(activity_embedding) > 1:
        activity_embedding = all_but_the_top(
          activity_embedding,
          n_principal_components=self.n_principal_components)
      self.embeddings.append(activity_embedding)
    # alignment using the 'Procrustes Orthogonal' transformation
    for k in range(1, len(self.embeddings)):
      self.embeddings[k-1], self.embeddings[k], _ = procrustes(
        self.embeddings[k-1], self.embeddings[k])
  
  def get_dynamic_embeddings(self, activity: str, time_slice_idx: int = -1) -> ArrayLike:
    activity_idx = self.act2idx[activity]
    if time_slice_idx != -1:
      return self.embeddings[time_slice_idx][activity_idx]
    return np.array([self.embeddings[t][activity_idx] for t in range(len(self.embeddings))])
  
  def k_nearest(self, activity: str, k: int = 5, time_slice_idx: int = -1) -> np.ndarray:
    activity_idx = self.act2idx[activity]
    if time_slice_idx != -1:
      act_emb = self.embeddings[time_slice_idx][activity_idx]
      sims = sim(act_emb.reshape(1, -1), self.embeddings[time_slice_idx]).reshape(-1)
      arg = np.argsort(sims)[-k - 1 : -1][::-1]
      neighbors = np.array([(self.idx2act[ind], sims[ind]) for ind in arg])
      return neighbors
    else:
      neighbors = np.array([self.k_nearest(activity, k, t)
                            for t in range(len(self.embeddings))])
      return neighbors


@dataclass(frozen=True)
class Acceleration:
  accelerator: Accelerator = None
  steps: ty.List[tuple] = field(default_factory=list)


@dataclass(frozen=True)
class TrainingReport:
  metrics: pd.DataFrame = None
  timeline_slices: list = field(default_factory=list)
  # NOTE: Keep this around to avoid 
  # passing unnecessary params to plot_activity_landscape
  timeline_slice_models: list = field(default_factory=list)
  diachronic_model: AlignedW2V = None
  
  def plot_activity_embeddings(
    self,
    # e.g., week
    time_slice: int,
    # activity name to its abbreviation
    act2abbr: dict,
    # plot options
    **kwargs) -> None:
    if len(act2abbr) == 0:
      raise ValueError("act2abbr is empty")
    
    activities = np.array([a for a in act2abbr])
    time_slice_idx = self.timeline_slices.index(time_slice)
    
    if self.diachronic_model is None:
      activity_model = self.timeline_slice_models[self.timeline_slices.index(time_slice)]
    else:
      activity_model = self.diachronic_model
      
    coordinates = [(lbl, x, y) for lbl, x, y in get_annotated_coordinates_from_model(
      activities, activity_model, 
      act2abbr, time_slice_idx=time_slice_idx)]
    plot_dynamic_activity_embeddings(coordinates, **kwargs)


def set_random_seed(seed):
  """Set random seed for reproducibility."""
  accelerate_utils.set_seed(seed)


def get_dataloaders_per_time_slice(
  train_test_data_by_time_slice: ty.List[tuple], 
  timeline_slices: ArrayLike, 
  batch_size: int = 10) -> ty.List[tuple]:

  if isinstance(timeline_slices, np.ndarray):
    timeline_slices = timeline_slices.tolist()

  train_test_dataloaders_by_time_slice = []
  for time_slice in timeline_slices:
    train_w, test_w = train_test_data_by_time_slice[timeline_slices.index(time_slice)]

    train_dataset = SkipgramData(train_w)
    test_dataset  = SkipgramData(test_w)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    train_test_dataloaders_by_time_slice.append((train_data_loader, test_data_loader))
    
  # return data snapshots wrapped in torch dataloaders
  return train_test_dataloaders_by_time_slice


def accelerate_model(
  train_test_dataloaders: ty.List[tuple],
  timeline_slices: ArrayLike, 
  act2idx: dict, 
  embedding_size: int = 120, 
  learn_rate: float = 0.001,
  accelerator: Accelerator = Accelerator()) -> Acceleration:
  
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
  
  return Acceleration(accelerator, accelerations)


def train_fn(model, train_data_loader, optimizer, criterion, epoch, accelerator, suffix='...'):
  model.train()
  fin_loss = 0.0
  tk = tqdm(
    train_data_loader,
    desc = f"EPOCH [TRAIN] {epoch + 1} [WEEK] {suffix}",
    disable=not accelerator.is_local_main_process,
  )

  total_samples = None
  for t, data in enumerate(tk):
    epoch_loss = 0
    
    if total_samples is None:
      total_samples = len(data[0])
    
    for (x, y) in zip(data[0], data[1]):
      optimizer.zero_grad()
      out = model(x)
      # print(out.shape, y.shape)
      loss = criterion(out, y)
      accelerator.backward(loss)
      optimizer.step()
      epoch_loss += loss.item()
      
      tk.set_postfix({
          "loss": "%.6f" % float(epoch_loss / (t + 1)),
          "LR": optimizer.param_groups[0]["lr"],
      })
    fin_loss += (epoch_loss * total_samples)
  

  return fin_loss / (len(train_data_loader) * total_samples), optimizer.param_groups[0]["lr"]


def test_fn(model, test_data_loader, criterion, epoch, accelerator, suffix='...'):
  model.eval()
  fin_loss = 0.0
  correct_ct = 0
  tk = tqdm(
    test_data_loader,
    desc = f"EPOCH [TEST] {epoch + 1} [WEEK] {suffix}",
    disable=not accelerator.is_local_main_process,
  )

  total_samples = None
  with torch.no_grad():
    for t, data in enumerate(tk):
      epoch_loss = 0
      if total_samples is None:
        total_samples = len(data[0])
      for (x, y) in zip(data[0], data[1]):
        out = model(x)
        loss = criterion(out, y)
        epoch_loss += loss.item()

        _, predicted = torch.max(out, 1)
        if predicted[0] == y[0]:
          correct_ct += 1
        

        tk.set_postfix({
            "loss": "%.6f" % float(epoch_loss / (t + 1)),
            "acc": "%.6f" % float((correct_ct / (t + 1)) * 1.0 * total_samples),
        })
      fin_loss += (epoch_loss * total_samples)
    
    return fin_loss / (len(test_data_loader) * total_samples), (correct_ct/(len(test_data_loader) * total_samples))*100


def learn_dynamic_activity_model(
  acceleration_config: Acceleration, 
  timeline_slices: ArrayLike,
  act2idx: dict, idx2act: dict,
  epochs: int,
  post_process_models: bool = True,
  n_principal_components: int = 10) -> ty.Tuple[AlignedW2V, TrainingReport]:
  
  accelerations = acceleration_config.steps
  accelerator = acceleration_config.accelerator
  
  if isinstance(timeline_slices, np.ndarray):
    timeline_slices = timeline_slices.tolist()

  # NOTE: If we have, for each time_slice (e.g., week), a stored
  # trained model (e.g., pickled model in some file), then we should load
  # it before we start their training. This includes its report, which
  # we can save in a separate csv file.
  
  # The following code learns a new model for each time_slice.
  timeline_slice_models = []
  metrics = []
  for time_slice in timeline_slices:
    model, optimizer, train_data, test_data = accelerations[timeline_slices.index(time_slice)]
    for epoch in range(epochs):
      criterion = nn.CrossEntropyLoss()
      
      avg_loss_train_week, lr = train_fn(
        model, train_data,
        optimizer, criterion,
        epoch, accelerator,
        suffix=f"{time_slice}...")
      avg_loss_eval_week, acc = test_fn(
        model, test_data,
        criterion, epoch,
        accelerator, suffix=f"{time_slice}...")
      
      metrics += [{
        'Epoch': epoch,
        'AverageLossTrain': avg_loss_train_week,
        'AverageLossEval': avg_loss_eval_week, 
        'LearningRate': lr, 
        'Accuracy': acc}]
    timeline_slice_models += [model]
  
  # Aligned activity models learned for each time_slice.
  dynamic_model = AlignedW2V(
    # activity vocabulary
    act2idx=act2idx,
    # inverse activity vocabulary
    idx2act=idx2act, 
    post_process_models=post_process_models,
    n_principal_components=n_principal_components)
  
  dynamic_model.fit(timeline_slices, timeline_slice_models)
  
  report = TrainingReport(
    metrics=pd.DataFrame.from_dict(metrics),
    timeline_slices=timeline_slices,
    timeline_slice_models=timeline_slice_models,
    diachronic_model=dynamic_model)
  
  return dynamic_model, report


def send_to_tensor(ctx, ctx2idx: dict) -> torch.Tensor:
  """Send data to tensor."""
  indices = [ctx2idx[w] for w in ctx]
  tensor = torch.tensor(indices, dtype=torch.long)
  return tensor


def generate_train_test_data(
  skipgrams_at_time_slice: ArrayLike, 
  act2idx: dict, 
  n_iters: int = 100, 
  test_size: float = 0.3) -> ty.Tuple[ArrayLike, ArrayLike]:
  X = skipgrams_at_time_slice[:]
  
  for _ in range(n_iters):
    X = np.random.permutation(X)
  
  X = [list(s) for s in X]
  X = [send_to_tensor(x, act2idx) for x in X]
  X = [X[i * 2: (i + 1) * 2] for i in range(len(X))]
  
  samples = []
  for xx in X:
    if len(xx) == 0:
      continue
    elif len(xx) == 1:
      samples.append((xx[0], xx[0]))
    else:
      samples.append((xx[0], xx[1]))

  if len(samples) == 1:
    # Instead of ignoring this record, we return the same
    # sample as train, test data (no need for sklearn's train_test_split)
    return samples, samples

  train, test = train_test_split(samples, test_size=test_size)
  return train, test


def generate_random_train_test_data(
  skipgrams: ArrayLike,
  txt2dict: dict,
  test_size: float = 0.3,
  sample_size: int = 1) -> ty.Tuple[ArrayLike, ArrayLike]:

  # NOTE: DO NOT USE THIS ONE. IT IS NOT WORKING PROPERLY.
  # USE generate_train_test_data INSTEAD.
  
  pivot = len(skipgrams) - int(len(skipgrams) * test_size)
  indices = multidimensional_shifting(len(skipgrams), sample_size, skipgrams).T[0]
  training_idx, test_idx = indices[:pivot], indices[pivot:]
  training, test = skipgrams[training_idx,:], skipgrams[test_idx,:]
  
  training = [(send_to_tensor(x, txt2dict), send_to_tensor(y, txt2dict))
              for x,y in training]
  test = [(send_to_tensor(x, txt2dict), send_to_tensor(y, txt2dict))
          for x,y in test]
  
  return training, test


def get_train_test_data_per_period(
  sliced_skipgrams: ArrayLike, 
  time_slices: ArrayLike,
  act2idx: dict,
  progress_bar: bool = False) -> ArrayLike:
  
  the_console = new_quiet_console()
  if progress_bar:
    the_console = stderr
  
  if isinstance(time_slices, np.ndarray):
    time_slices = time_slices.tolist()
  
  train_test_data = []
  with new_progress_display(the_console) as progress:
    task = progress.add_task("Collecting data in time slices ...", total=len(sliced_skipgrams))
    for time_slice in time_slices:
      train_, test_ = generate_train_test_data(sliced_skipgrams[time_slice], act2idx, n_iters=10)
      # train_, test_ = generate_random_train_test_data(
      #   sliced_skipgrams[time_slices.index(time_slice)], txt2dict)
      train_test_data.append((train_, test_))
      progress.update(task, advance=1)
  
  return np.array(train_test_data)


def get_tensor_data(tensor: torch.Tensor) -> ArrayLike:
  if torch.cuda.is_available():
    return tensor.detach().cpu().numpy()
  else:
    return tensor.detach().data.numpy()


def get_annotated_coordinates_from_model(
  activities: ArrayLike, 
  trained_model: ty.Any, 
  act2abbr: dict, 
  time_slice_idx: int = -1) -> ty.Iterator[tuple]:
  """Get coordinates from tensor."""
  for activity in activities:
    if isinstance(trained_model, AlignedW2V):
      embedding = trained_model.get_dynamic_embedding(activity, time_slice_idx)
      x = embedding_data[0]
      y = embedding_data[1]
    else:
      embedding = trained_model.get_embedding(activity)
      embedding_data = get_tensor_data(embedding)
      x = embedding_data[0][0]
      y = embedding_data[0][1]
    yield act2abbr[activity], x, y


def take_n_from_data_loader(n: int, t: int, train_test_dataloaders_by_time_slice: ty.List[tuple]) -> list:
  return take(n, train_test_dataloaders_by_time_slice[t][0])


if __name__ == "__main__":
  pass
