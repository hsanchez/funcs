#!/usr/bin/env python

import typing as ty

from pinstall import install as install_package

try:
  import numpy as np
except ImportError:
  install_package('numpy', True)
  import numpy as np


def tensor_fusion(h_x: np.array, h_y: np.array) -> np.ndarray:
  """Computes the tensor fusion based on the recommendation of 
  Morency et al., Tutorial on Multimodal Machine Learning

  - current implementation: Assumes 1-D vectors!
  """

  # add 1 to h_x and h_1 at the end and beginning, respectively
  h_x = np.append(h_x, 1)
  h_y = np.concatenate(([1], h_y))

  # add new axis to h_x and h_y
  ## if h_x is (N, ), convert it to (N, 1)
  h_x_1 = np.expand_dims(h_x, axis=1)

  ## if h_y is (M, ), convert it to (1, M)
  h_y_1 = np.expand_dims(h_y, axis=0)

  # compute the Kronecker product on h_x_1 and h_y_1
  K_xy = np.kron(h_x_1, h_y_1)

  return K_xy
  
def vector_fusion_3D(x: np.array, y: np.array, z: np.array) -> np.ndarray:    
  # add 1 at the beginning of each 1-D vector
  h_x = np.concatenate((np.ones(1), x), axis=0)
  h_y = np.concatenate((np.ones(1), y), axis=0)
  h_z = np.concatenate((np.ones(1), z), axis=0)

  # add additional axis needed for 3D vector fusion
  ## (x_shape, 1, 1)
  h_x = h_x[..., np.newaxis, np.newaxis]

  ## (1, y_shape, 1)
  h_y = h_y[np.newaxis, ..., np.newaxis]

  ## (1, 1, z_shape)
  h_z = h_z[np.newaxis, np.newaxis, ...]

  # Kronecker product, equivalent to h_x * h_y * h_z
  h_m = np.kron(h_x, np.kron(h_y, h_z))

  return h_m


def get_rolling_window(array: ty.Union[list, np.ndarray], window_size: int) -> np.ndarray:
  if isinstance(array, list):
    array = np.array(array)
  return np.lib.stride_tricks.sliding_window_view(array, window_size)


def drop_missing_vals(x: np.ndarray) -> np.ndarray:
  if x.ndim > 1:
    return x[~np.isnan(x).any(axis=1)]
  return x[~np.isnan(x)]


if __name__ == "__main__":
  pass
