#!/usr/bin/env python

import typing as ty
from collections import namedtuple
from numbers import Number

from .console import new_progress_display, stderr
from .modules import install as install_package

try:
  import numpy as np
except ImportError:
  install_package('numpy', True)
  import numpy as np

try:
  from scipy.spatial import procrustes
  from scipy.spatial.distance import cdist, pdist, squareform
except ImportError:
  install_package('scipy')
  from scipy.spatial import procrustes
  from scipy.spatial.distance import cdist, pdist, squareform

try:
  from sklearn.decomposition import PCA
except ImportError:
  install_package('scikit-learn')
  from sklearn.decomposition import PCA
  

NumberLike = ty.Union[Number, ty.Iterable[Number]]


def sigmoid(x: NumberLike) -> NumberLike:
  # to avoid RuntimeWarning: Overflow encountered in exp warning
  x = np.float128(x)
  return 1.0 / (1.0 + np.exp(-x))


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
  """Summarize a sequence of vectors by taking the weighted similarity of their vectors.
  Args:
    vecs: a sequence of vectors
    alpha: the similarity mixing factor
  Returns:
    a vector
  """
  U = np.array(vecs_seq[0])
  S = np.add.reduce([cosine_similarity(U, vec_j) * np.array(vec_j) for vec_j in vecs_seq[1:]])
  N = np.sum([cosine_similarity(U, vec_j) for vec_j in vecs_seq[1:]])
  
  return U * alpha + (1 - alpha) * (S / N)


def apply_seq_kernel(vec: ty.Union[ty.List[float], np.ndarray], kernel_radius: int = 1, epsilon: float = 0.01) -> np.ndarray:
  # apply Gaussian kernel to a vector
  if isinstance(vec, list):
    vec = np.array(vec)

  kernels = squareform(pdist(np.expand_dims(np.arange(vec.size), axis=-1)))
  
  # Calculate graph kernels with cutoff of epsilon at the kernel_radius.
  kernels = np.exp((kernels * np.log(epsilon)) / kernel_radius)
  kernels[kernels < epsilon] = 0

  # Normalize kernels by dividing by row sums.
  kernels = kernels / np.expand_dims(np.sum(kernels, axis=-1), axis=-1)
  
  # Updates sequence embeddings using kernel
  c_emb_prime = np.dot(kernels, vec)
  
  return c_emb_prime


def build_distance_matrix(vec_i: np.ndarray, vec_j: np.ndarray) -> np.ndarray:
  return 1.0 - cdist(vec_i, vec_j, metric='cosine')


def build_scoring_matrix(a: np.ndarray, w_i: float = 1.0, w_j: float = 1.0, epsilon: float = 0.01) -> np.ndarray:
  # Pad distance matrix
  sa = np.pad(a, ((1,0),(1,0)), 'constant', constant_values = 0)
  
  # Calculate gap weight kernels
  dims = a.shape
  w_i_ = [w_i * np.exp((i * np.log(epsilon)) / dims[0]) for i in reversed(range(dims[0] + 1))]
  w_j_ = [w_j * np.exp((j * np.log(epsilon)) / dims[1]) for j in reversed(range(dims[1] + 1))]
  
  # Updates scoring matrix according to policy
  for i in range(1, dims[0] + 1):
    for j in range(1, dims[1] + 1):
      inputs = [
        # Top Left + Bottom Right
        (sa[i, j] + sa[i - 1,j - 1]),
        # Max of all previous values in column - column gap weight
        np.max(sa[:i,j]) - w_i_[i - np.argmax(sa[:i, j])],
        # Max of all previous values in row - row gap weight
        np.max(sa[i,:j]) - w_j_[j - np.argmax(sa[i, :j])],
        # Zero
        0]
      sa[i, j] = np.max(inputs)
  return sa


def perform_traceback(sa: np.ndarray, k: ty.Optional[int] = 100) -> np.ndarray:
  # Sort scoring matrix values in descending order; Save coordinates in look up table.
  sorted_args = np.argsort(sa.flatten())[::-1]
  coords = [(i, j) for i in range(sa.shape[0]) for j in range(sa.shape[1])]
  
  # Perform traceback until all coords have been visited
  tracebacks = []
  seen = []
  route = []
  
  for idx in sorted_args:
    # matrix indices
    i, j = coords[idx]
    
    flag = True
    score = sa[i, j]

    while(flag):
      # Route connects to other traceback
      if (i, j) in seen:
        tracebacks.append([route, (i, j)])
        route = []
        break
      
      route.append((i, j))
      seen.append((i, j))
      
      # Route terminates at zero
      if sa[i, j] == 0:
        tracebacks.append([route, []])
        route = []

      # Select path direction
      kernel = [sa[i - 1, j], sa[i,j - 1], sa[i - 1, j - 1], sa[i, j]]
      m = np.argmax(kernel)
      
      # Move to next gap
      if m == 0:
        # Terminate route if score is less than gap value
        if score > sa[i - 1, j]:
          i -= 1
          score += sa[i, j]
        else:
          tracebacks.append([route, []])
          route = []
          break
      elif m == 1:
        # Terminate route if score is less than gap value
        if score > sa[i, j - 1]:
          j -= 1
          score += sa[i, j]
        else:
          tracebacks.append([route, []])
          route = []
          break
      # Move to next hit
      elif m in [2, 3]:
        i -= 1
        j -= 1
        score += sa[i, j]
      
      # Stop at zero or if route is too long
      if i < 0 or j < 0:
        break


  # Return alignments with length greater than 1 in order as they are found.
  if k is None:
    k = len(tracebacks)

  alignments = [] # a collection of index tuples
  for _ in tracebacks:
    # check length of routes
    if len(_[0]) > 1:
      r = [(i - 1, j - 1) for (i, j) in _[0]]
      alignments.append(r[:-1])
    if len(alignments) == k:
      break

  return alignments


def score_alignment(alignment: np.ndarray, seq_x: np.ndarray, seq_y: np.ndarray, k: int) -> float:
  # Find gaps and hits, and gather feature vectors
  temp_i = []
  temp_j = []
  
  i = -1
  j = -1
  segment_a = []
  segment_b = []

  for each_idx_pair in alignment:
    if each_idx_pair[0] != i:
      temp_i.append(1)
      i = each_idx_pair[0]
    else: temp_i.append(0.0)
    if each_idx_pair[1] != j:
      temp_j.append(1)
      j = each_idx_pair[1]
    else: temp_j.append(0.0)
    segment_a.append(seq_x[each_idx_pair[0]])
    segment_b.append(seq_y[each_idx_pair[1]])
  
  # Calculate similarity score
  mask = np.array(temp_i) * np.array(temp_j)
  similarity = 2 - cdist(segment_a, segment_b, 'cosine').diagonal()
  score = (similarity * mask) / (2 * len(alignment)) * (np.sum(mask) / len(seq_y)) * k * len(seq_y)
  
  if not score or np.isnan(score):
    stderr.print('alignment score is NaN', style='yellow')
    score = 0.0

  return score[0]


def process_alignments(
  top_alignments: np.ndarray, 
  top_scores: np.ndarray, 
  input_vecs: ty.List[np.ndarray],
  vec_index: dict) -> list:
  
  assert len(input_vecs) == 2, 'Only two sequences can be aligned at a time.'
  v_i = input_vecs[0]
  v_j = input_vecs[1]
  Alignment = namedtuple('Alignment', ['score', 'X', 'Y'])
  
  results = []
  with new_progress_display(console=stderr) as progress:
    task = progress.add_task('Process alignments ...', total=len(top_alignments))
    for i, a in enumerate(top_alignments):
      X = []
      Y = []
      l = -1
      r = -1
      for each_idx_pair in reversed(a):
        # process the left index
        if each_idx_pair[0] != l:
          if vec_index:
            cands_l = vec_index.get(tuple(v_i[each_idx_pair[0]]), None)
            cands_l = [tpl[1] for tpl in cands_l if tpl[0] == 0]
            X += [cands_l[0] if len(cands_l) > 0 else None]
          else:
            X.append(v_i[each_idx_pair[0]])
          l = each_idx_pair[0]
        else:
          X.append('GAP')

        # process the right index
        if each_idx_pair[1] != r:
          if vec_index:
            cands_j = vec_index.get(tuple(v_j[each_idx_pair[1]]), None)
            cands_j = [tpl[1] for tpl in cands_j if tpl[0] == 1]
            Y += [cands_j[0] if len(cands_j) > 0 else None]
          else:
            Y.append(v_j[each_idx_pair[1]])
          r = each_idx_pair[1]
        else:
          Y.append('GAP')
      results += [Alignment(top_scores[i], np.array(X), np.array(Y))]
      progress.update(task, advance=1)
  return results


if __name__ == "__main__":
  pass
