#!/usr/bin/env python

import re
import typing as ty

import numpy as np

from .pinstall import install as install_package

try:
  from sparse_dot_topn import awesome_cossim_topn
except:
  install_package('sparse_dot_topn')
  from sparse_dot_topn import awesome_cossim_topn

try:
  from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
  install_package('scikit-learn')
  from sklearn.feature_extraction.text import TfidfVectorizer


def _ngrams_analyzer(string):
  string = re.sub(r'[,-./]', r'', string)
  ngrams = zip(*[string[i:] for i in range(5)])  # N-Gram length is 5
  ngrams_arr = [''.join(ngram) for ngram in ngrams]
  return ngrams_arr


def _get_cosine_matrix(tf_idf_matrix, vals):
  # The arguments for awesome_cossim_topn are as follows:
  ### 1. Our TF-IDF matrix
  ### 2. Our TF-IDF matrix transposed (allowing us to build a pairwise cosine matrix)
  ### 3. A top_n filter, which allows us to filter the number of matches returned, which isn't useful for our purposes
  ### 4. This is our similarity threshold. Only values over 0.8 will be returned

  cosine_matrix = awesome_cossim_topn(
    tf_idf_matrix,
    tf_idf_matrix.transpose(),
    vals.size,
    0.8)
  return cosine_matrix

def _find_group(row, col, group_lookup):
  # If either the row or the col string have already been given
  # a group, return that group. Otherwise return none
  if row in group_lookup:
    return group_lookup[row]
  elif col in group_lookup:
    return group_lookup[col]
  else:
    return None
  

def _add_pair_to_lookup(row, col, group_lookup):
  # in this function we'll add both the row and the col to the lookup
  group = _find_group(row, col, group_lookup)  # first, see if one has already been added
  if group is not None:
    # if we already know the group, make sure both row and col are in lookup
    group_lookup[row] = group
    group_lookup[col] = group
  else:
    # if we get here, we need to add a new group.
    # The name is arbitrary, so just make it the row
    group_lookup[row] = row
    group_lookup[col] = row


def deduplicate(names: ty.Set[str]) -> ty.Set[str]:
  names = np.array(names)
  vectorizer = TfidfVectorizer(analyzer=_ngrams_analyzer)
  tf_idf_matrix  = vectorizer.fit_transform(names)
  cosine_matrix = _get_cosine_matrix(tf_idf_matrix, names)
  
  # Build a coordinate matrix
  coo_matrix = cosine_matrix.tocoo()
  
  group_lookup = {}
  # for each row and column in coo_matrix
  # if they're not the same string add them to the group lookup
  for row, col in zip(coo_matrix.row, coo_matrix.col):
    if row != col:
      _add_pair_to_lookup(names[row], names[col], group_lookup)
    else:
      group_lookup[names[row]] = names[col]
  return set([v for _, v in group_lookup.items()])


if __name__ == "__main__":
  pass
