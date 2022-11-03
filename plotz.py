#!/usr/bin/env python

import typing as ty

import numpy as np
import pandas as pd

from .pinstall import install as install_package


def build_plotly_browser_state(build_html: ty.Any) -> ty.Any:
  return build_html('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-5.11.0.min.js?noext',
            },
          });
        </script>
        ''')


def plot_correlation_heatmap(
  input_df: pd.DataFrame,
  **kwargs) -> pd.DataFrame:
  
  pyplot_module = kwargs.get('pyplot_module', None)
  seaborn_module = kwargs.get('seaborn_module', None)
  threshold = kwargs.get('threshold', 0)
  figsize = kwargs.get('figsize', (10, 8))
  title = kwargs.get('title', "Column Correlation Heatmap")
  
  if pyplot_module is None:
    try:
      import matplotlib.pyplot as plt
    except ImportError:
      install_package('matplotlib')
      import matplotlib.pyplot as plt
  else:
    plt = pyplot_module
  
  if seaborn_module is None:
    try:
      import seaborn as sns
    except ImportError:
      install_package('seaborn')
      import seaborn as sns
  else:
    sns = seaborn_module
    
  corr = input_df.corr()
  corr = corr.where(np.abs(corr) > threshold, 0)

  # Generate a mask for the upper triangle
  mask = np.zeros_like(corr, dtype=np.bool)
  mask[np.triu_indices_from(mask)] = True

  # Set up the matplotlib figure
  f, ax = pyplot_module.subplots(figsize=figsize)

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
