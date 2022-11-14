#!/usr/bin/env python

import re
import typing as ty

import numpy as np
import pandas as pd

from .arrays import get_mode_in_array
from .console import new_progress_display, new_quiet_console, stderr
from .modules import install as install_package

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


def split_txt(txt : str, upper: bool = False) -> str:
  joined_txt = '\n'.join(t for t in txt.split())
  if upper:
    return joined_txt.upper()
  return joined_txt


def abbreviate_txt(txt : str, upper: bool = False, sep: str = None) -> str:
  def split_flavor(txt, sep = None):
    if sep is None:
      return txt.split()
    else:
      return txt.split(sep)
  joined_first_chars = ''.join(t[0] for t in split_flavor(txt, sep))
  if upper:
    return joined_first_chars.upper()
  return joined_first_chars


def build_txt_indices(input_df: pd.DataFrame, target_column: str) -> ty.Tuple[dict, dict]:
  txt2abbr = {}
  abbr2txt = {}
  for val in input_df[target_column].values:
    abbr = abbreviate_txt(val)
    txt2abbr[val] = abbr
    txts = abbr2txt.get(abbr, [])
    txts.append(val)
    abbr2txt[abbr] = txts
  # txt resolution using mode
  for key in abbr2txt.keys():
    m = get_mode_in_array(abbr2txt[key])
    if m is not None:
      abbr2txt[key] = m
    else:
      # pick first one
      abbr2txt[key] = abbr2txt[key][0]
  return txt2abbr, abbr2txt


def generate_skipgrams(
  input_df: pd.DataFrame, 
  vocab: ty.Iterable, 
  window_date_offset: pd.DateOffset,
  target_col: str,
  progress_bar: bool = True,
  datetime_column: str = 'sent_time') -> ty.List[tuple]:
  
  the_console = stderr
  if not progress_bar:
    the_console = new_quiet_console()
  
  vocab_size = sum(1 for _ in vocab)
  skipgrams = []
  with new_progress_display(console = the_console) as progress:
    task = progress.add_task("Generating skipgrams ...", total=vocab_size)
    for w in vocab:
      # collect all records from the ACTIVITY_DF where triplet_one == a.
      # note that an activity with the triple_one format is an activity relabeled 
      # with **three** elements: 
      #   original activity name + project name + email thread topology 
      res_a = input_df.loc[input_df[target_col] == w]
      # iterate over all res_a records and collect all records 
      # that are within a time window
      for _, row_a in res_a.iterrows():
        s_time = row_a[datetime_column] - window_date_offset
        e_time = row_a[datetime_column] + window_date_offset
        
        res_row_a = input_df.loc[(input_df[datetime_column] >= s_time) & (input_df[datetime_column] <= e_time)]
        # FIX: fixes a recursive dependency issue
        # res_row_a = get_records_in_time_window(input_df, datetime_column, s_time, e_time)
        
        w_skipgrams = [(w, act_x) for act_x in res_row_a[target_col].tolist()]
        skipgrams.extend(w_skipgrams)
        
      progress.update(task, advance=1)
  return skipgrams


if __name__ == "__main__":
  pass
