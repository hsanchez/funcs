#!/usr/bin/env python

import typing as ty

from .pinstall import install as install_package

try:
  import numpy as np
except ImportError:
  install_package('numpy', True)
  import numpy as np

try:
  from scipy.spatial import procrustes
except ImportError:
  install_package('scipy')
  from scipy.spatial import procrustes

try:
  from sklearn.decomposition import PCA
except ImportError:
  install_package('scikit-learn')
  from sklearn.decomposition import PCA


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


def get_mode_in_array(array: np.ndarray) -> ty.Optional[ty.Any]:
  if not array or array.size == 0:
    return None
  vals, counts = np.unique(array, return_counts=True)
  idx = np.argmax(counts)
  return vals[idx]


def init_U_and_V_embeddings(vocab_size: int, timeline_size: int, rank: int = 50) -> ty.Tuple[np.ndarray, np.ndarray]:
  """Initialize variables (embeddings) for the model:
    U, V = init_vars(vocab_size=288, timeline_size=len(timeline))
    U.shape, V.shape
  """
  U, V = [], []
  U.append(np.random.randn(vocab_size,rank) / np.sqrt(rank))
  V.append(np.random.randn(vocab_size,rank) / np.sqrt(rank))
  
  for t in range(1, timeline_size):
    U.append(U[0].copy())
    V.append(V[0].copy())
  
  return np.array(U), np.array(V)


def all_but_the_top(v, n_principal_components=10):
  # thx to https://gist.github.com/lgalke/febaaa1313d9c11f3bc8240defed8390
  # All-but-the-Top: Simple and Effective Postprocessing for Word Representations
  # Paper: https://arxiv.org/abs/1702.01417
  
  # 1. Subtract mean vector
  v_tilde = v - np.mean(v, axis=0)
  
  # 2. Compute the first 'n' principal components
  #    on centered embedding vectors
  u = PCA(n_components=n_principal_components).fit(v_tilde).components_  # [n_principal_components, emb_size]
  
  # Subtract first 'n' principal components
  # [vocab_size, emb_size] @ [emb_size, D] @ [n_principal_components, emb_size] -> [vocab_size, emb_size]
  return v_tilde - (v @ u.T @ u)


def procrustes_orthogonal(embeddings: np.ndarray) -> np.ndarray:
  for k in range(1, len(embeddings)):
    embeddings[k-1], embeddings[k], _ = procrustes(
    embeddings[k-1], embeddings[k])
  return embeddings


def make_matrices_same_size(Y_tgt: np.ndarray, U_src: np.ndarray) -> np.ndarray:
  U_src_exp = np.zeros((Y_tgt.shape[0], Y_tgt.shape[1]), dtype=complex)
  U_src_exp[:U_src.shape[0], :U_src.shape[1]] = U_src
  return U_src_exp


def drop_zero_rows(input_array: np.ndarray) -> np.ndarray:
    """Remove all rows that are composed of all zeroes.
    Args:
      input_array: ndarray to sanitize
    Returns:
      The resulting array after removal of all zero rows.
    """
    input_array = input_array[~np.all(input_array == 0, axis=1)]

    return input_array


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
  return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def summarize_vectors_sequence(vecs_seq: np.ndarray, alpha: float = 0.5) -> np.ndarray:
  """Summarize a sequence of vectors by taking the weighted average of the vectors.
  Args:
    vecs: a sequence of vectors
    alpha: the weight of the first vector in the sequence
  Returns:
    a vector
  """
  return np.average(vecs_seq, axis=0, weights=np.arange(alpha, 1.0, (1.0 - alpha) / len(vecs_seq)))


def summarize_vectors_sequence_by_weighted_mixing(vecs_seq: np.ndarray, alpha: float = 0.5) -> np.ndarray:
  U = np.array(vecs_seq[0])

  S = [cosine_similarity(U, vec_j) * np.array(vec_j) for vec_j in vecs_seq[1:]]
  S = np.add.reduce(S)
  N = np.sum([cosine_similarity(U, vec_j) for vec_j in vecs_seq[1:]])
  
  return U * alpha + (1 - alpha) * (S / N)


if __name__ == "__main__":
  pass
