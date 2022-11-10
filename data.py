#!/usr/bin/env python

import itertools
import pathlib as pl
import typing as ty
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from .arrays import ArrayLike
from .common import resolve_path, with_status
from .console import new_progress_display, quiet_stderr, stderr, stdout
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


# Based on the persuasion strategies in 
# Wang, X., Shi, W., Kim, R., Oh, Y., Yang, S., Zhang, J., & Yu, Z. (2019). 
# Persuasion for good: Towards a personalized persuasive dialogue system for social good.
# arXiv preprint [arXiv:1906.06725](https://arxiv.org/abs/1906.06725) 
IDX2PERSUASION = dict({0: 'task-related-inquiry',
                       1: 'credibility-appeal',
                       2: 'logical-appeal', 
                       3: 'personal-related-inquiry', 
                       4: 'source-related-inquiry', 
                       5: 'donation-information',
                       6: 'foot-in-the-door',
                       7: 'emotion-appeal',
                       8: 'self-modeling',
                       9: 'personal-story',
                       10: 'Unknown'})


COLUMN_RENAMES = {
  'message_exper': 'sender_commitment',
  'commit_exper': 'sender_experience',
  'is_patch_churn': 'is_patch_update',
  'is_first_patch_thread': 'is_first_patch_in_thread'}


# Define columns of interest
RELEVANT_FEATURES = [
  # Developers characteristics
  'sender_commitment',
  'sender_experience',

  # Exposition
  'fkre_score', 
  'fkgl_score',
  'word_cnt',
  'sentence_cnt',
  'is_persuasive',

  # Email characteristics
  'is_patch_email', 
  'is_first_patch_in_thread', 
  'sent_time',
  'received_time', 
  'is_quickly_replied',

  # Patches characteristics
  'is_patch_update',
  'is_bug_fix',
  'is_new_feature',
  'is_accepted_patch',
  'is_accepted_commit',
]


@dataclass(frozen=True)
class ProcessingResults:
  raw_data: pd.DataFrame = None
  unnormed_data: pd.DataFrame = None
  metrics: pd.DataFrame = None


def _check_input_dataframe(input_df: pd.DataFrame) -> None:
  # Empty Dataframe check
  assert input_df.shape[0] > 0 and input_df.shape[1] > 0 , "DataFrame is Empty"
  # duplicate columns check
  assert len(input_df.columns.values)==len(set(input_df.columns.values)) , "DataFrame has duplicate columns"


def build_single_row_dataframe(data: dict) -> pd.DataFrame:
  return pd.DataFrame(data, index=[0])


def build_multi_index_dataframe(data: ArrayLike, multi_index_df: pd.DataFrame, columns: ArrayLike) -> pd.DataFrame:
  _check_input_dataframe(multi_index_df)
  
  return pd.DataFrame(
    data = data, 
    index=pd.MultiIndex.from_frame(multi_index_df),
    columns=columns)


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


def drop_records_match_condition(input_df: pd.DataFrame, condition: ty.Callable[[pd.Series], bool] = None, indices_only: bool = False, inplace: bool = False) -> pd.DataFrame:
  if condition is None:
    return input_df
  if indices_only:
    return input_df.drop(input_df[condition(input_df)].index, inplace=inplace)
  return input_df.drop(condition(input_df), inplace=inplace)


def find_binary_columns(input_df: pd.DataFrame) -> list:
  _check_input_dataframe(input_df)
  return input_df.columns[input_df.isin([0, 1]).all()].tolist()


def find_boolean_columns(input_df: pd.DataFrame) -> list:
  _check_input_dataframe(input_df)
  return input_df.columns[input_df.dtypes == bool].tolist()


def detect_nan_columns(input_df: pd.DataFrame) -> list:
  _check_input_dataframe(input_df)
  return input_df.columns[input_df.isnull().any()].tolist()


def standardize_dataframe(input_df: pd.DataFrame, add_gaussian_noise: bool = False) -> pd.DataFrame:
  # Standardize values to have mean zero and unit variance.
  # (We will use these standardized data in factor analysis.)
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


def datetime_column_to_timestamp(input_df: pd.DataFrame, column: str) -> pd.DataFrame:
  _check_input_dataframe(input_df)
  # Turn datetime values into Unix second time
  # unix sec time; thx to https://stackoverflow.com/questions/54312802/
  input_df[column] = pd.to_datetime(input_df[column]).view(np.int64) // 10 ** 9
  return input_df


def normalize_column(
  input_df: pd.DataFrame,
  column: str,
  is_datetime: bool = False,
  min_norm: int = 0,
  max_norm: int = 1) -> pd.DataFrame:
  """
  Normalize the column values to a range of [min_norm, max_norm]
  """
  _check_input_dataframe(input_df)
  df = input_df.copy()
  if is_datetime:
    # TODO(anyone): check whether we need the 10 ** 9 factor
    df[column] = pd.to_datetime(df[column]).view(np.int64)
  else:
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min()) * (max_norm - min_norm) + min_norm
  return df


def normalize_columns(
  input_df: pd.DataFrame,
  min_norm: int = 0,
  max_norm: int = 1) -> pd.DataFrame:
  """
  Normalize the column values to a range of [min_norm, max_norm]
  """
  _check_input_dataframe(input_df)
  df = input_df.copy()
  
  datetime_columns = set(df.select_dtypes(include=[np.datetime64]).columns.to_list())
  for col in df.columns:
    df = normalize_column(
      df,
      col,
      is_datetime=(col in datetime_columns),
      min_norm=min_norm,
      max_norm=max_norm)
  return df


def describe_timeline(time_frame_skipgrams: np.ndarray, time_periods_in_window_range: ty.Union[np.ndarray, list]) -> pd.DataFrame:
  """Prints the distribution of the skipgrams in the time frame"""
  if isinstance(time_periods_in_window_range, np.ndarray):
    time_periods_in_window_range = time_periods_in_window_range.tolist()
  
  # Count the number of skipgrams in each time frame
  time_frame_skipgrams_counts = Counter(time_frame_skipgrams)
  # Print the distribution
  stdout.print("Distribution of skipgrams in the time frame")
  
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


def fast_read_and_append(file_path: str, chunksize: int = None, factor: int = 4, fullsize: float = 1e9, dtype: ty.Any = None, console: ty.Any = None, separator: str = ',') -> pd.DataFrame:
  import math

  import psutil as ps

  # in chunk reading be careful as pandas might infer a columns dtype 
  # as different for diff chunk. As such specifying a dtype while
  # reading by giving params to read_csv maybe better. Label encoding
  # will fail if half the rows for same column is int and rest are str.
  # In case of that already happened then 
  #         df_test["publisherId"] = df_test["publisherId"].apply(str)
  resolved_file_path = pl.Path(resolve_path(file_path))
  
  if chunksize is None:
    try:
      # estimate the available memory for the dataframe chunks
      chunksize = (
        ps.virtual_memory().available // (pd.read_csv(
          str(resolved_file_path), 
          sep=separator, nrows=1).memory_usage(deep=True).sum() * factor))
    except Exception:
      chunksize = 1000
      stderr.print(f"[yellow]Failed to estimate chunksize for file: {resolved_file_path}")
      stderr.print(f"[yellow]Set default chunksize to: {chunksize}")

  df = pd.DataFrame()
  total_needed_iters = math.ceil(fullsize / chunksize)
  with new_progress_display(console) as progress:
    task = progress.add_task(f"Reading {resolved_file_path.name} ...", total=total_needed_iters)
    for x in pd.read_csv(str(resolved_file_path), sep=separator, chunksize=chunksize, dtype=dtype):
      df = df.append(x)
      df = pd.concat([df, x], ignore_index=True)
      progress.update(task, advance=1)
    return df


def ordinal_encode(input_df: pd.DataFrame, cols: ty.List[str], encoding_view: bool = False) -> pd.DataFrame:
  """
  Perform ordinal encoding transformation on the specified columns 
  in the dataframe. The expected shape is 2D.
  """
  _check_input_dataframe(input_df)
  df_ord = input_df.copy()
  if not cols or len(cols) == 0:
    cols = df_ord.columns.to_list()
  enc = OrdinalEncoder()
  df_ord[cols] = enc.fit_transform(df_ord[cols])
  if encoding_view:
    # encoding view
    return input_df.assign(rating_enc=df_ord)
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



def process_and_clean_dataframe(
  input_df: pd.DataFrame,
  features: ty.List[str] = RELEVANT_FEATURES,
  idx2persuasion: dict = IDX2PERSUASION, 
  name2name: dict = COLUMN_RENAMES,
  quiet: bool = False) -> pd.DataFrame:
  _check_input_dataframe(input_df)
  
  the_console = stderr
  if quiet:
    the_console = quiet_stderr
  
  df = input_df.copy()
  
  @with_status(console=the_console, prefix="Generate features")
  def process_relevant_columns(data: pd.DataFrame) -> ty.Tuple[pd.DataFrame, ProcessingResults]:
    report = ProcessingResults()
    # Prior analysis on the LKML confirmed that patch emails
    # tend to get a response within 3.5 hrs (on average).
    # For sake of simplicity, we used 4 hrs instead of 3.5 hrs. 
    # 4 * 60 * 60. Any response within 4 hrs is considered as 'quickly responded'.
    four_h_lapse = 14400
    data['is_quickly_replied'] = data['time_lapse'].apply(
      lambda x: 1 if (x <= four_h_lapse and x != -1) else 0)
    
    persuasion_set = idx2persuasion.keys() - {10}
    # Is Patch Email persuasive?
    data['is_persuasive'] = data['persuasion'].apply(
      lambda x: 1 if (x in persuasion_set) else 0)
    drop_columns_safely(data, ['persuasion'], True)
    
    # Filter all rows for which the developer's
    # word count on their emails is greater than or equal to 50
    # df.drop(df[df['word_cnt'] < 50].index, inplace = True)
    drop_records_match_condition(data, lambda x: x.word_cnt < 50, True, True)
    
    # Turn datetime values into Unix second time
    # unix sec time; thx to https://stackoverflow.com/questions/54312802/
    # or to https://stackoverflow.com/questions/40881876/

    data = datetime_column_to_timestamp(data, 'sent_time')
    data = datetime_column_to_timestamp(data, 'received_time')
    
    # Renames certain columns
    rename_columns_in_dataframe(data, name2name, inplace=True)
    
    reduced_data = data[features].copy()
    # Computes verbosity of the patch email
    reduced_data['verbosity'] = reduced_data['word_cnt'] / reduced_data['sentence_cnt']
    ## Replace NaN cases with Zero and the drop word and sentence counts
    reduced_data.fillna(value={'verbosity': 0}, inplace=True)
    drop_columns_safely(reduced_data, ['word_cnt', 'sentence_cnt'], inplace=True)

    (no_rows, no_feats) = reduced_data.shape
    no_binary_feats = len(find_binary_columns(reduced_data))
    
    metrics = {
      'RecordCount' : no_rows,
      'FeatureCount' : no_feats,
      'NumericFeatures' : no_feats - no_binary_feats,
      'BinaryFeatures' : no_binary_feats}
    
    metrics = build_single_row_dataframe(metrics)
    report = replace(report, raw_data=data, unnormed_data=reduced_data, metrics=metrics)
    
    reduced_data_norm = standardize_dataframe(reduced_data, add_gaussian_noise=True)
    
    return (reduced_data_norm, report)
  
  df, report = process_relevant_columns(df)
  
  @with_status(console=the_console, prefix="Standardize features")
  def standardize_with_noise(data: pd.DataFrame) -> pd.DataFrame:
    return standardize_dataframe(data, add_gaussian_noise=True)
  
  # Standardize values to have mean zero and unit variance.
  df = standardize_with_noise(df)
  
  # returns interest_df_norm, report
  return df, report


if __name__ == "__main__":
  pass
