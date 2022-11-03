#!/usr/bin/env python

from .pinstall import install as install_package

try:
  import cufflinks as cf
except ImportError:
  install_package('cufflinks')
  import cufflinks as cf

try:
  import seaborn as sns
except ImportError:
  install_package('seaborn')
  import seaborn as sns

try:
  import plotly
except ImportError:
  install_package('plotly', True)
  import plotly

import typing as ty

import numpy as np
import pandas as pd
from plotly.offline import init_notebook_mode


def init_plotly_notebook_mode(go_offline: bool = True, connected: bool = False) -> None:
  if go_offline:
    cf.go_offline()
  init_notebook_mode(connected=connected)


def configure_plotly_browser_state(render_html: ty.Any, display_fn: ty.Callable[..., ty.Any]):
  display_fn(render_html('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-5.11.0.min.js?noext',
            },
          });
        </script>
        '''))


def plot_correlation_heatmap(
  input_df: pd.DataFrame,
  plt: ty.Any,
  title: str = "Column Correlation Heatmap", 
  threshold: int = 0, figsize: ty.Tuple[int, int] = (10, 8)) -> pd.DataFrame:
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


if __name__ == "__main__":
  pass
