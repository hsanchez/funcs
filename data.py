#!/usr/bin/env python

import itertools
import pathlib as pl
import typing as ty

from .arrays import ArrayLike
from .modules import install as install_package

try:
  from sklearn.preprocessing import OrdinalEncoder
except ImportError:
  install_package('scikit-learn')
  from sklearn.preprocessing import OrdinalEncoder

try:
  import pandas as pd
except ImportError:
  install_package('pandas', True)
  import pandas as pd


try:
  import scipy.cluster.hierarchy as shc
  from scipy.cluster.hierarchy import cophenet
  from scipy.spatial.distance import euclidean, pdist # computing the distance
except ImportError:
  install_package('scipy')
  import scipy.cluster.hierarchy as shc
  from scipy.cluster.hierarchy import cophenet
  from scipy.spatial.distance import euclidean, pdist # computing the distance

from collections import Counter, namedtuple

import numpy as np

from .common import resolve_path
from .console import new_progress_display, stdout, stderr


ClusterReport = namedtuple('ClusterReport', ['data', 'cophentic_corr', 'cluster_labels'])


def _check_input_dataframe(input_df: pd.DataFrame) -> None:
  # Empty Dataframe check
  assert input_df.shape[0] > 0 and input_df.shape[1] > 0 , "DataFrame is Empty"
  # duplicate columns check
  assert len(input_df.columns.values)==len(set(input_df.columns.values)) , "DataFrame has duplicate columns"


def get_column_names(input_df: pd.DataFrame, sorted: bool = True) -> ty.List[ty.Any]:
  if sorted:
    return list(np.sort(input_df.columns.values.tolist()))
  return input_df.columns.values.tolist()


def count_nulls_in_dataframe(input_df: pd.DataFrame) -> pd.DataFrame:
  """Missing value count per column grouped by column name"""
  
  df_t = pd.DataFrame(input_df.isnull().sum()).rename(columns={0: "count"})
  df_t["percent_null"] = 100.0 * df_t["count"] / input_df.shape[0]
  df_t.index.names = ["Column"]
  
  return df_t.sort_values("percent_null", ascending=False)


def get_column_datatypes(input_df: pd.DataFrame) -> pd.DataFrame:
  _check_input_dataframe(input_df)
  dtype = {}
  for idx in input_df.columns.values:
    dt = input_df[idx].dtype
    dtype[idx] = dt
  ctr = pd.DataFrame([dtype]).T
  ctr = ctr.rename(columns={0: 'datatype'})
  ctr.index.names = ["Column"]
  return ctr


def find_correlated_pairs(input_df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
  _check_input_dataframe(input_df)
  corr = input_df.corr()
  corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
  corr = corr.stack().reset_index()
  corr.columns = ['var1', 'var2', 'corr']
  corr = corr.loc[corr['corr'].abs() > threshold]
  return corr


def drop_columns_safely(input_df: pd.DataFrame, columns: list, inplace: bool = False) -> ty.Optional[pd.DataFrame]:
  _check_input_dataframe(input_df)
  intersected_columns = list(set(input_df.columns.values).intersection(set(columns)))
  return input_df.drop(intersected_columns, axis=1, inplace=inplace)


def drop_indices_match_condition(input_df: pd.DataFrame, condition: ty.Callable[[pd.Series], bool] = None, inplace: bool = False) -> ty.Optional[pd.DataFrame]:
  if condition is None:
    return input_df
  return input_df.drop(input_df[condition(input_df)].index, inplace=inplace)


def drop_records_match_condition(input_df: pd.DataFrame, condition: ty.Callable[[pd.Series], bool] = None) -> pd.DataFrame:
  if condition is None:
    return input_df
  return input_df.drop[condition(input_df)]


def find_zero_one_columns(input_df: pd.DataFrame) -> list:
  _check_input_dataframe(input_df)
  return input_df.columns[input_df.isin([0, 1]).all()].tolist()


def find_boolean_columns(input_df: pd.DataFrame) -> list:
  _check_input_dataframe(input_df)
  return input_df.columns[input_df.dtypes == bool].tolist()


def detect_nan_columns(input_df: pd.DataFrame) -> list:
  _check_input_dataframe(input_df)
  return input_df.columns[input_df.isnull().any()].tolist()


def standardize_dataframe(input_df: pd.DataFrame, add_gaussian_noise: bool = False) -> pd.DataFrame:
  _check_input_dataframe(input_df)
  
  df = input_df.copy()
  if add_gaussian_noise:
    # We first add Gaussian noise to all the values in the dataframe
    (no_rows, no_feats) = df.shape
    mu, sigma = 0, 0.1
    # creating a noise with the same dimension as the df
    noise = np.random.normal(mu, sigma, (no_rows, no_feats))
    noisy_df = df + noise
    df = noisy_df
  
  for col in df.columns:
    # zero mean: remove the average
    # unit variance: divide by the standard deviation
    df[col] = (df[col] - df[col].mean()) / df[col].std()
  return df


def get_records_in_time_window(
  input_df: pd.DataFrame, 
  start_time_column: str, 
  start_time: np.datetime64, 
  end_time: np.datetime64, 
  end_time_column: str = None) -> pd.DataFrame:
    """Obtain dataframe records that fall within a user-defined time window

    Args:
      input_df: input dataframe
      time_column: dataframe column storing the time data serving as search filter
      start_time: time window start time
      end_time: time window end time

    Returns:
      The resulting subset dataframe
    """

    assert start_time <= end_time
    
    if end_time_column is None:
      end_time_column = start_time_column

    tmp_df = input_df.loc[(input_df[start_time_column] >= start_time) & (input_df[end_time_column] <= end_time)]

    return tmp_df


def normalize_column(
  input_df: pd.DataFrame,
  is_datetime: bool = False,
  min_norm: int = 0,
  max_norm: int = 1) -> pd.DataFrame:
  """
  Normalize the column values to a range of [min_norm, max_norm]
  """
  _check_input_dataframe(input_df)
  df = input_df.copy()
  if is_datetime:
    df[col] = pd.to_datetime(df[col]).astype(np.int64)
  for col in df.columns:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min()) * (max_norm - min_norm) + min_norm
  return df


def describe_timeline(time_frame_skipgrams: np.ndarray, time_periods_in_window_range: ty.Union[np.ndarray, list]) -> pd.DataFrame:
  """Prints the distribution of the skipgrams in the time frame"""
  if isinstance(time_periods_in_window_range, np.ndarray):
    time_periods_in_window_range = time_periods_in_window_range.tolist()
  
  # Count the number of skipgrams in each time frame
  time_frame_skipgrams_counts = Counter(time_frame_skipgrams)
  # Print the distribution
  print("Distribution of skipgrams in the time frame")
  
  data_for_dataframe = {}
  for time_frame, skipgrams_count in time_frame_skipgrams_counts.items():
    data_for_dataframe[time_frame] = skipgrams_count
    time_period = time_periods_in_window_range[time_frame]
    stdout.print(f"Time period {time_period}: {skipgrams_count} skipgrams")

  return pd.DataFrame(data_for_dataframe, index=[0])


def get_unique_column_values(input_df: pd.DataFrame, target_column: str) -> np.ndarray:
  _check_input_dataframe(input_df)
  return input_df[target_column].unique()


def generate_skipgrams(
  input_df: pd.DataFrame, 
  vocab: ty.Iterable, 
  window_date_offset: pd.DateOffset,
  target_col: str, 
  datetime_column: str = 'sent_time') -> ty.List[tuple]:
  
  _check_input_dataframe(input_df)
  vocab_size = sum(1 for _ in vocab)
  skipgrams = []
  with new_progress_display(console = stdout) as progress:
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
        
        # get all members in time window
        res_row_a = get_records_in_time_window(input_df, datetime_column, s_time, e_time)
        
        w_skipgrams = [(w, act_x) for act_x in res_row_a[target_col].tolist()]
        skipgrams.extend(w_skipgrams)
        
      progress.update(task, advance=1)
  return skipgrams


def timeline_slicing(
  timeline: pd.DataFrame,
  target_column: str,
  datetime_col: str = 'sent_time',
  window_size: int = 4,
  by_period: str = 'week') -> ty.Tuple[np.ndarray, np.ndarray]:
  
  if by_period not in ['year', 'week', 'day']:
    raise ValueError(f"Invalid value for by_period: {by_period}")

  def get_time_period_series(df: pd.DataFrame, bp: str, col: str) -> ty.Any:
    if bp == 'year':
      return df[col].dt.isocalendar().year
    elif bp == 'week':
      return df[col].dt.isocalendar().week
    elif bp == 'day':
      return df[col].dt.isocalendar().day
    else:
      raise ValueError(f"Invalid value for time_period: {bp}")
    

  data = []
  # window size
  ws = pd.DateOffset(hours=window_size)
  
  by_period_series = get_time_period_series(timeline, by_period, datetime_col)
  start_period = by_period_series.min()
  end_period = by_period_series.max() + 1
  
  if by_period == 'year':
    ws = pd.DateOffset(years=window_size)
  
  tp_in_window_range = range(start_period, end_period)  
  for time_period in tp_in_window_range:
    # 1. get data-frame partition
    tp_df = timeline.loc[by_period_series == time_period]
    # 2. get unique activities list for activity column
    tp_unique_vals = get_unique_column_values(input_df=tp_df, target_column=target_column)
    # 3. get skipgrams for the current time period
    skipgrams_by_tp = generate_skipgrams(
      tp_df, tp_unique_vals, ws,
      target_col=target_column, datetime_column=datetime_col)
    # 4. add the list to final_data, [[], [], []]
    data.append(skipgrams_by_tp)
  return np.array(data, dtype=object), np.array(list(tp_in_window_range))


def build_cooccur_matrix(skipgrams: np.ndarray, activities: ty.List[str], all_time_periods: bool = False) -> np.ndarray:
  def build_matrix(x: np.ndarray, acts: ty.List[str], size: int, period: int = -1):
    mat = np.zeros([size, size], dtype=int)
    if period == -1:
      # process all time periods
      for j in range(x.shape[0]):
        for i in x[j]:
          mat[acts.index(i[0])][acts.index(i[1])] += 1
    else:
      # process a specific time period
      for i in skipgrams[period]:
        mat[acts.index(i[0])][acts.index(i[1])] += 1
    return mat

  cooccur = []
  if all_time_periods:
    cooccur.append(build_matrix(skipgrams, activities, len(activities)))
  else:
    for t in range(skipgrams.shape[0]):
      cooccur.append(build_matrix(skipgrams, activities, len(activities), period = t))
  return np.array(cooccur)


def fast_read_and_append(file_path: str, chunksize: int, fullsize: float = 1e9, dtype: ty.Any = None, console: ty.Any = None, separator: str = ',') -> pd.DataFrame:
  import math

  # in chunk reading be careful as pandas might infer a columns dtype 
  # as different for diff chunk. As such specifying a dtype while
  # reading by giving params to read_csv maybe better. Label encoding
  # will fail if half the rows for same column is int and rest are str.
  # In case of that already happened then 
  #         df_test["publisherId"] = df_test["publisherId"].apply(str)
  resolved_file_path = pl.Path(resolve_path(file_path))
  df = pd.DataFrame()
  total_needed_iters = math.ceil(fullsize / chunksize)
  with new_progress_display(console) as progress:
    task = progress.add_task(f"Reading {resolved_file_path.name} ...", total=total_needed_iters)
    for x in pd.read_csv(str(resolved_file_path), sep=separator, chunksize=chunksize, dtype=dtype):
      df = df.append(x)
      df = pd.concat([df, x], ignore_index=True)
      progress.update(task, advance=1)
    return df


def ordinal_encode(df: pd.DataFrame, cols: ty.List[str], encoding_view: bool = False) -> pd.DataFrame:
  """
  Perform ordinal encoding transformation on the specified columns 
  in the dataframe. The expected shape is 2D.
  """
  df_ord = df.copy()
  if not cols or len(cols) == 0:
    cols = df_ord.columns.to_list()
  enc = OrdinalEncoder()
  df_ord[cols] = enc.fit_transform(df_ord[cols])
  if encoding_view:
    # encoding view
    return df.assign(rating_enc=df_ord)
  return df_ord

def get_pairwise_co_occurrence(array_of_arrays: ty.List[list], items_taken_together: int = 2) -> pd.DataFrame:
  counter = Counter()
  for v in array_of_arrays:
    permuted_values = list(itertools.combinations(v, items_taken_together))
    counter.update(permuted_values)
  # The key in the dict being a list cannot be possible unless it's converted to a string.
  co_oc = pd.DataFrame(
    np.array([[key,value] for key,value in counter.items()]),
    columns=['items_taken_together','frequency'])
  co_oc['frequency'] = co_oc['frequency'].astype(int)
  co_oc = co_oc[co_oc['frequency'] > 0]
  co_oc = co_oc.sort_values(['frequency'], ascending=False)
  return co_oc


def get_records_match_condition(input_df: pd.DataFrame, condition: ty.Callable[[pd.Series], bool] = None) -> pd.DataFrame:
  # e.g., get_records_match_condition(df, df.month == 'January')
  if condition is None:
    return input_df
  return input_df.loc[condition(input_df)]


def select_columns_subset_from_dataframe(input_df: pd.DataFrame, columns: ty.List[str]) -> pd.DataFrame:
    """Selects a subset of columns from a given input dataframe.

    Args:
      input_df: input dataframe
      columns: subset of columns to select

    Returns:
      The resulting subset dataframe

    Raises:
      LookupError: If columns do not exist in the input_df 
    """

    # check if columns exist in the input dataframe
    if set(columns).issubset(input_df.columns):
        result_df = input_df[columns]
    else:
        raise LookupError(f"Input columns: {columns} do not exist in the dataframe!")
    
    return result_df


def cluster_data(
  input_df: pd.DataFrame,
  callback: ty.Callable[[ArrayLike, ty.Any], None] = None,
  **kwargs) -> ClusterReport:
  _check_input_dataframe(input_df)
  
  Z = shc.linkage(input_df, method='ward')
  if not Z:
    raise ValueError("Z is empty!")
  
  c, _ = cophenet(Z, pdist(input_df))
  cophenetic_corr_coeff = round(c, 2)
  
  cluster_labels = []
  flat_option = kwargs.get('flat_option', False)
  if flat_option:
    no_clusters = kwargs.get('no_clusters', 5)
    criterion = kwargs.get('criterion', 'maxclust')
    cluster_labels = shc.fcluster(Z, no_clusters, criterion=criterion)
  
  if callback:
    callback(Z, **kwargs)
  
  return ClusterReport(Z, cophenetic_corr_coeff, cluster_labels)


def compute_role_change_intensity(
  input_df: pd.DataFrame,
  target_column: str,
  cluster_centers: ArrayLike,
  dist_fn: ty.Callable[..., float] = euclidean) -> float:
  _check_input_dataframe(input_df)
  RCI = 0.0
  with new_progress_display(console=stderr) as progress:
    task = progress.add_task("Computing RCI ...", total=len(input_df))
    for (_,x),(_,y) in zip(input_df[:-1].iterrows(), input_df[1:].iterrows()):
      R_cur = y[target_column].astype(np.int64)
      R_prev = x[target_column].astype(np.int64)
      RCI += (dist_fn(cluster_centers[R_cur], cluster_centers[R_prev]))
      progress.update(task, advance=1)
  return round(np.log10(RCI), 2)


def rename_columns_in_dataframe(
  input_df: pd.DataFrame, 
  name2name: ty.Dict[str, str], 
  inplace: bool = True) -> ty.Optional[pd.DataFrame]:
  """Renames columns in a given dataframe.

  Args:
    input_df: input dataframe
    columns: dictionary of columns to rename
    inplace: whether to rename columns in place or not
  """
  _check_input_dataframe(input_df)
  return input_df.rename(columns=name2name, inplace=inplace)

if __name__ == "__main__":
  pass
