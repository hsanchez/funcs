import pathlib as pl
import typing as ty

from sklearn.preprocessing import OrdinalEncoder

import numpy as np
import pandas as pd

from .colabs import resolve_path
from .console import progress_display


def _check_input_dataframe(input_df: pd.DataFrame) -> None:
  # Empty Dataframe check
  assert input_df.shape[0]>0 and input_df.shape[1]>0 , "DataFrame is Empty"
  # duplicate columns check
  assert len(input_df.columns.values)==len(set(input_df.columns.values)) , "DataFrame has duplicate columns"


def __particular_values_per_column(input_df: pd.DataFrame, values: list) -> pd.DataFrame:
  _check_input_dataframe(input_df)
  counts = {}
  for idx in input_df.columns.values:
    cnt = np.sum(input_df[idx].isin(values).values)
    counts[idx] = cnt
  ctr = pd.DataFrame([counts]).T
  ctr_2 = ctr.rename(columns={0: '# Values as %s'%values})
  return ctr_2


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


def count_distinct_values(input_df: pd.DataFrame) -> pd.DataFrame:
  _check_input_dataframe(input_df)
  unique_counts = {}
  for idx in input_df.columns.values:
    cnt = input_df[idx].nunique()
    unique_counts[idx] = cnt
  unique_ctr = pd.DataFrame([unique_counts]).T
  unique_ctr = unique_ctr.rename(columns={0: 'count'})
  unique_ctr.index.names = ["Column"]
  return unique_ctr.sort_values("count", ascending=False)


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


def get_most_common_value(input_df: pd.DataFrame) -> pd.DataFrame:
  # for simple cases use df.mode().T
  _check_input_dataframe(input_df)
  total_rows = input_df.shape[0]
  columns, modes, counts = list(), list(), list()
  for column in input_df.columns:
    count = input_df[column].value_counts().max()
    mode = input_df[column].value_counts().index[0]
    columns.append(column)
    modes.append(mode)
    counts.append(count)
  description = pd.DataFrame(
    index=columns,
    data={"MostFrequentValue": modes, 
          "MostFrequentValueCount": counts})
  description["MostFrequentValue %"] = 100 * description["MostFrequentValueCount"] / total_rows
  description.index.names = ["Column"]
  return description


def describe_dataframe(input_df: pd.DataFrame) -> pd.DataFrame:
  _check_input_dataframe(input_df)

  numeric_only_df = input_df.select_dtypes(np.number)
  mis_val = numeric_only_df.isnull().sum()
  mis_val_percent = 100 * mis_val / len(numeric_only_df)
  particular_ctr = __particular_values_per_column(numeric_only_df, [0])
  unique_ctr = count_distinct_values(numeric_only_df)
  statistical_summary = numeric_only_df.describe().T
  datatypes = get_column_datatypes(numeric_only_df)
  most_common_values = get_most_common_value(numeric_only_df)
  skewed = pd.DataFrame(numeric_only_df.skew()).rename(columns={0: 'skew'})
  mis_val_table = pd.concat([
    mis_val,
    mis_val_percent,
    unique_ctr,
    particular_ctr,
    datatypes,
    skewed,
    statistical_summary,
    most_common_values], axis=1)
  mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% missing of Total Values'})

  return mis_val_table_ren_columns


def find_correlated_pairs(input_df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
  _check_input_dataframe(input_df)
  corr = input_df.corr()
  corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
  corr = corr.stack().reset_index()
  corr.columns = ['var1', 'var2', 'corr']
  corr = corr.loc[corr['corr'].abs() > threshold]
  return corr


def drop_columns_safely(input_df: pd.DataFrame, columns: list, inplace: bool = False) -> pd.DataFrame:
  _check_input_dataframe(input_df)
  intersected_columns = list(set(input_df.columns.values).intersection(set(columns)))
  return input_df.drop(intersected_columns, axis=1, inplace=True)


def remove_correlated_pairs(input_df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
  _check_input_dataframe(input_df)
  nulls_df = count_nulls_in_dataframe(input_df).T
  corr_pairs = find_correlated_pairs(input_df, threshold)
  dropped_cols = set()
  for (p1, p2) in corr_pairs:
    if p1 not in dropped_cols and p2 not in dropped_cols:
      if nulls_df[p1].values[0] > nulls_df[p2].values[0]:
        dropped_cols.add(p2)
      else:
        dropped_cols.add(p1)
  dropped_cols = list(np.sort(list(dropped_cols)))
  new_df = drop_columns_safely(input_df, dropped_cols)
  return new_df, dropped_cols


def detect_nan_columns(input_df: pd.DataFrame) -> list:
  _check_input_dataframe(input_df)
  return input_df.columns[input_df.isnull().any()].tolist()


def fast_read_and_append(file_path: str, chunksize: int, fullsize: float = 1e9, dtype: ty.Any = None) -> pd.DataFrame:
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
  with progress_display as progress:
    task = progress.add_task(f"Reading {resolved_file_path.name} ...", total=total_needed_iters)
    for x in pd.read_csv(str(resolved_file_path), chunksize=chunksize, dtype=dtype):
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
  enc = OrdinalEncoder()
  df_ord[cols] = enc.fit_transform(df_ord[cols])
  if encoding_view:
    # encoding view
    return df.assign(rating_enc=df_ord)
  return df_ord


if __name__ == "__main__":
  pass
