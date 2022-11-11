#!/usr/bin/env python

import typing as ty

import numpy as np
import pandas as pd

from .arrays import ArrayLike
from .modules import import_module
from .modules import install as install_package
from .nlp import split_txt

try:
  import scipy.cluster.hierarchy as shc
except ImportError:
  install_package('scipy')
  import scipy.cluster.hierarchy as shc

try:
  import plotly.express as px
except ImportError:
  install_package('scipy')
  import plotly.express as px


def plot_RCI_distribution(input_df: pd.DataFrame, **kwargs) -> None:
  seaborn_module = kwargs.get('seaborn_module', None)
  if seaborn_module is None:
    sns = import_module('seaborn')
  else:
    sns = seaborn_module
  
  sns.histplot(input_df, **kwargs)
  

def plot_activity_distribution(input_df: pd.DataFrame, **kwargs) -> None:
  # see https://plotly.com/python/bar-charts/
  plotly_module = kwargs.get('plotly_module', None)
  if plotly_module is None:
    px = import_module('seaborn')
  else:
    px = plotly_module
  
  fig = px.bar(input_df, **kwargs)
  fig.show()


def scree_plot(input_df: pd.DataFrame, eigenvalues: ArrayLike, **kwargs) -> None:
  plt_module = kwargs.get('pyplot_module', None)
  if plt_module is None:
    plt = import_module('matplotlib.pyplot', 'matplotlib')
  else:
    plt = plt_module
  
  figsize = kwargs.get('figsize', (8, 8))
  plt.figure(figsize=figsize)
  plt.scatter(range(1, input_df.shape[1] + 1), eigenvalues)
  plt.plot(range(1, input_df.shape[1] + 1), eigenvalues)
  plt.title('Scree Plot')
  plt.xlabel('Factors')
  plt.ylabel('Eigenvalue')
  plt.axhline(y=1.0, color='r', linestyle='-')
  plt.grid()
  plt.show()
  
  
def plot_factors_heatmap(input_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
  plt_module = kwargs.get('pyplot_module', None)
  seaborn_module = kwargs.get('seaborn_module', None)
  figsize = kwargs.get('figsize', (10, 8))
  title = kwargs.get('title', "Factors to Characteristics Heatmap")

  if plt_module is None:
    plt = import_module('matplotlib.pyplot', 'matplotlib')
  else:
    plt = plt_module
    
  if seaborn_module is None:
    sns = import_module('seaborn')
  else:
    sns = seaborn_module

  # Generate a custom diverging colormap
  cmap = sns.diverging_palette(240, 10, as_cmap=True)
  
  # get correlation matrix plot for loadings
  plt.figure(figsize=figsize)
  plt.title(title)
  ax = sns.heatmap(
    input_df, cmap=cmap,
    vmax=1.0, vmin=-1.0,
    cbar_kws={"shrink": .8},
    center=0, square=True, 
    linewidths=.5, annot=True, fmt='.2f')
  plt.show()

  return input_df


def plot_column_correlation_heatmap(input_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
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
  f, ax = plt.subplots(figsize=figsize)

  # Generate a custom diverging colormap
  cmap = sns.diverging_palette(240, 10, as_cmap=True)

  # Draw the heatmap with the mask and correct aspect ratio
  sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, cbar_kws={"shrink": .8}, center=0,
              square=True, linewidths=.5, annot=True, fmt='.2f')
  plt.title(f"{title}")
  plt.show()
  return corr


def find_no_clusters_by_elbow_plot(k, data: ArrayLike, **kwargs) -> None:
  plt_module = kwargs.get('pyplot_module', None)
  if plt_module is None:
    plt = import_module('matplotlib.pyplot', 'matplotlib')
  else:
    plt = plt_module
  
  if plt:
    figsize = kwargs.get('figsize', (10, 5))
    plt.figure(figsize=figsize)
    plt.title('Optimal number of cluster')
    plt.xlabel('Number of cluster (k)')
    plt.ylabel('Total intra-cluster variation')
    plt.plot(range(1, k+1), data, marker = "x")
    plt.show()
    

def find_no_clusters_by_dist_growth_acceleration_plot(Z_input: ArrayLike, quiet: bool = False, **kwargs) -> ty.Optional[int]:
  plt_module = kwargs.get('pyplot_module', None)
  if plt_module is None:
    plt = import_module('matplotlib.pyplot', 'matplotlib')
  else:
    plt = plt_module
  
  figsize = kwargs.get('figsize', (10, 5))
  
  last = Z_input[-10:, 2]
  last_rev = last[::-1]
  indices = np.arange(1, len(last) + 1)

  plt.figure(figsize=figsize)
  plt.title('Optimal number of cluster')
  plt.xlabel('Number of cluster')
  if not quiet:
    plt.plot(indices, last_rev, marker = "o", label="distance")

  acceleration = np.diff(last, 2)  # 2nd derivative of the distances
  acceleration_reversed = acceleration[::-1]
  if not quiet:
    plt.plot(indices[:-2] + 1, acceleration_reversed, marker = "x", label = "2nd derivative of distance growth")
    plt.legend()
    plt.show()
  k = acceleration_reversed.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
  return k


def radar_plot(input_df: pd.DataFrame, **kwargs) -> None:
  plt_module = kwargs.get('pyplot_module', None)
  if plt_module is None:
    plt = import_module('matplotlib.pyplot', 'matplotlib')
  else:
    plt = plt_module
  
  input_df_T = input_df.T
  labels = list(input_df_T.index)
  
  figsize = kwargs.get('figsize', (1000/96, 1000/96))
  dpi = kwargs.get('dpi', 96)
  prefix_title = kwargs.get('prefix_title', 'Role')
  suptitle = kwargs.get('suptitle', 'Activity space characteristics of roles in LKML')
  suptitle_weight = kwargs.get('subtitle_weight', 'bold')
  wspace = kwargs.get('wspace', 1.)

  # initialize the figure
  fig = plt.figure(figsize=figsize, dpi=dpi)
  fig.suptitle(suptitle, weight=suptitle_weight)
  
  # prepare the grid
  fig.subplots_adjust(wspace=wspace)
  
  # Create a color palette and define text color:
  color_palette = plt.cm.get_cmap("Set2", len(labels))
  text_color = "#565656"
  
  def realign_polar_xticks(ax):
    for x, label in zip(ax.get_xticks(), ax.get_xticklabels()):
      if np.sin(x) > 0.1:
        label.set_horizontalalignment('left')
      if np.sin(x) < -0.1:
        label.set_horizontalalignment('right')

  def _make_spider(df, row, title, color, text_color):
    # number of variables (one per radar plot)
    categories = list(df.columns)
    categories = [split_txt(str(l), upper=True) for l in categories]
    N = len(categories)
    
    # calculate evenly-spaced axis angles
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # initialize the spider plot
    # TODO(HAS) try this one with 1, len(labels)
    # ax = plt.subplot(1, len(labels), row+1, polar=True)
    ax = plt.subplot(3, 3, row+1, polar=True,)
    
    # If you want the first axis to be on top:
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    ax.set_ylim(0, 1.)
    ax.set_yticks([])
    ax.xaxis.grid(linewidth=1)
    ax.yaxis.grid(linewidth=1)
    
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, size=8)
    
    # Draw ylabels
    ax.set_rlabel_position(180)
    realign_polar_xticks(ax)
    
    PAD = 0.05
    ax.text(0.05, 0 + PAD, "5%", size=8, color=text_color, fontname="DejaVu Sans")
    ax.text(0.05, 0.25 + PAD, "25%", size=8, color=text_color, fontname="DejaVu Sans")
    ax.text(0.05, 0.5 + PAD, "50%", size=8, color=text_color, fontname="DejaVu Sans")
    ax.text(0.05, 0.75 + PAD, "75%", size=8, color=text_color, fontname="DejaVu Sans")
    ax.text(0.05, 0.9 + PAD, "100%", size=8, color=text_color, fontname="DejaVu Sans")
    
    values  = df.loc[row].values.tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=.4)
    
    # Add a title (with title positioning using y, and loc in {right, center, left})
    plt.title(title, size=12, color=color, y=1.2, loc='center')
  
  for idx, row in enumerate(labels):
    _make_spider(
      df=input_df_T,
      row=idx, 
      title=f'\n{prefix_title} {row}',
      color=color_palette(row), 
      text_color=text_color)


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
