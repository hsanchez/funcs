#!/usr/bin/env python

import typing as ty

import numpy as np
import pandas as pd

from .arrays import ArrayLike
from .console import stderr
from .modules import import_module
from .modules import install as install_package

try:
  import scipy.cluster.hierarchy as shc
except ImportError:
  install_package('scipy')
  import scipy.cluster.hierarchy as shc


def plot_correlation_heatmap(input_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
  
  pyplot_module = kwargs.get('pyplot_module', None)
  seaborn_module = kwargs.get('seaborn_module', None)
  threshold = kwargs.get('threshold', 0)
  figsize = kwargs.get('figsize', (10, 8))
  title = kwargs.get('title', "Column Correlation Heatmap")
  
  if pyplot_module is None:
    plt = import_module('matplotlib.pyplot', 'matplotlib')
  else:
    plt = pyplot_module
  
  if seaborn_module is None:
    sns = import_module('seaborn')
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


def find_no_clusters_by_elbow_plot(k, data: ArrayLike, **kwargs) -> None:
  plt = kwargs.get('pyplot_module', None)
  if plt:
    figsize = kwargs.get('figsize', (10, 5))
    plt.figure(figsize=figsize)
    plt.title('Optimal number of cluster')
    plt.xlabel('Number of cluster (k)')
    plt.ylabel('Total intra-cluster variation')
    plt.plot(range(1, k+1), data, marker = "x")
    plt.show()
    

def find_no_clusters_by_dist_growth_acceleration_plot(Z_input: ArrayLike, **kwargs) -> ty.Optional[int]:
  plt = kwargs.get('pyplot_module', None)
  if plt is None:
    stderr.print('No pyplot module provided')
    return None
  
  figsize = kwargs.get('figsize', (10, 5))
  
  last = Z_input[-10:, 2]
  last_rev = last[::-1]
  idxs = np.arange(1, len(last) + 1)

  plt.figure(figsize=figsize)
  plt.title('Optimal number of cluster')
  plt.xlabel('Number of cluster')

  plt.plot(idxs, last_rev, marker = "o", label="distance")

  accele = np.diff(last, 2)  # 2nd derivative of the distances
  accele_rev = accele[::-1]
  plt.plot(idxs[:-2] + 1, accele_rev, marker = "x", label = "2nd derivative of distance growth")

  plt.legend()
  plt.show()
  k = accele_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
  return k


# thx to https://bit.ly/3siFoaZ
def make_dendrogram(*args, **kwargs):
  max_d = kwargs.pop('max_d', None)
  if max_d and 'color_threshold' not in kwargs:
    kwargs['color_threshold'] = max_d
  annotate_above = kwargs.pop('annotate_above', 0)
  
  pyplot_module = kwargs.get('pyplot_module', None)
  if pyplot_module is None:
    plt = import_module('matplotlib.pyplot', 'matplotlib')
  else:
    plt = pyplot_module

  ddata = shc.dendrogram(*args, **kwargs)

  if not kwargs.get('no_plot', False):
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('cluster size')
    plt.ylabel('distance')
    for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
      x = 0.5 * sum(i[1:3])
      y = d[1]
      if y > annotate_above:
        plt.plot(x, y, 'o', c=c)
        plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                     textcoords='offset points', 
                     va='top', ha='center')
    if max_d:
      plt.axhline(y=max_d, c='k')
  return ddata



if __name__ == "__main__":
  pass
