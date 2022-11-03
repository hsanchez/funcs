import itertools
import typing as ty
from collections import Counter

import cufflinks as cf
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import init_notebook_mode

import numpy as np
import pandas as pd


def init_plotly_notebook_mode(go_offline: bool = True, connected: bool = False) -> None:
  if go_offline:
    cf.go_offline()
  init_notebook_mode(connected=False)


def get_plotly_browser_state() -> ty.Any:
  import IPython

  return IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-1.5.1.min.js?noext',
            },
          });
        </script>
        ''')


def plot_correlation_heatmap(input_df: pd.DataFrame, title: str = "Column Correlation Heatmap", threshold: int = 0, figsize: ty.Tuple[int, int] = (10, 8)) -> pd.DataFrame:
  corr = input_df.corr()
  corr = corr.where(np.abs(corr) > threshold, 0)

  # Generate a mask for the upper triangle
  mask = np.zeros_like(corr, dtype=np.bool)
  mask[np.triu_indices_from(mask)] = True

  # Set up the matplotlib figure
  f, ax = plt.subplots(figsize=figsize)

  # Generate a custom diverging colormap
  cmap = sns.diverging_palette(240, 10, as_cmap=True)

  # Draw the heatmap with the mask and correct aspect ratio
  sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, cbar_kws={"shrink": .8}, center=0,
              square=True, linewidths=.5, annot=True, fmt='.2f')
  plt.title(f"{title}")
  plt.show()
  return corr


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


if __name__ == "__main__":
  pass
