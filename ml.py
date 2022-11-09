#!/usr/bin/env python

import typing as ty
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .arrays import ArrayLike
from .console import new_progress_display, stderr
from .data import _check_input_dataframe, build_single_row_dataframe, build_multi_index_dataframe
from .highlights import highlight_eigenvalues
from .modules import install as install_package

try:
  from factor_analyzer import FactorAnalyzer
  from factor_analyzer.factor_analyzer import (calculate_bartlett_sphericity,
                                               calculate_kmo)
except ImportError:
  install_package('factor_analyzer')
  from factor_analyzer import FactorAnalyzer
  from factor_analyzer.factor_analyzer import (calculate_bartlett_sphericity,
                                               calculate_kmo)

try:
  import scipy.cluster.hierarchy as shc
  from scipy.cluster.hierarchy import cophenet
  from scipy.spatial.distance import euclidean, pdist  # computing the distance
except ImportError:
  install_package('scipy')
  import scipy.cluster.hierarchy as shc
  from scipy.cluster.hierarchy import cophenet
  from scipy.spatial.distance import euclidean, pdist  # computing the distance


try:
  from sklearn.cluster import AgglomerativeClustering
except ImportError:
  install_package('scikit-learn')
  from sklearn.cluster import AgglomerativeClustering


@dataclass(frozen=True)
class ClusterReport:
  data: ArrayLike = field(default_factory=list)
  cophentic_corr: float = 0.0
  cluster_labels: ArrayLike = field(default_factory=list)


@dataclass
class FactorAnalysisReport:
  metrics: pd.DataFrame = None
  factor_scores: pd.DataFrame = None


def factor_analysis(
  # FYI: this is the interest_df_norm
  input_df: pd.DataFrame,
  k: int = 10,
  metrics_only: bool = False,
  multi_index_df: pd.DataFrame = None,
  index_columns: ArrayLike = None,
  summary_plot: bool = False,
  rotation: ty.Optional[str] = None,
  **kwargs) -> ty.Tuple[pd.DataFrame, FactorAnalysisReport]:
  
  _check_input_dataframe(input_df)
  
  # Metrics
  report = FactorAnalysisReport()
  
  chi_square_value, p_value = calculate_bartlett_sphericity(input_df)
  # kmo_all, kmo_model
  _, kmo_model = calculate_kmo(input_df)
  
  metrics = {
    'Chi_Square_Value' : round(chi_square_value, 2),
    'P_Value' : p_value,
    'Kaiser_Meyer_Olkin_Score' : round(kmo_model, 2)}
  report.metrics = build_single_row_dataframe(metrics)
  
  if metrics_only:
    # Return untouched input_df 
    # and its factor analysis report
    return input_df, report
  
  # Create factor analysis object and perform factor analysis
  fa = FactorAnalyzer(k, rotation=rotation)
  fa.fit(input_df)
  
  if rotation is None:
    # Check Eigenvalues
    ev, _ = fa.get_eigenvalues()
    factor_labels = ['Factor' + ' ' + str(i + 1) for i in range(len(input_df.columns))]
    eigenvalues_df = pd.DataFrame(data=ev, index=factor_labels, columns=["Eigenvalue"])
    
    if summary_plot:
      from .plots import plot_scree_plot
      # scree plot
      plot_scree_plot(input_df.copy(), eigenvalues_df['Eigenvalue'].values.tolist(), **kwargs)
    # eigenvalues_df.style.apply(highlight_eigenvalues, color='yellow')
    
    return eigenvalues_df, report
  else:
    # Get factor loadings
    loadings_matrix = fa.loadings_
    factor_labels = ['Factor' + ' ' + str(i + 1) for i in range(k)]
    
    # factors' loadings
    loadings_df = pd.DataFrame(
      data = loadings_matrix, 
      index = input_df.columns, 
      columns = factor_labels)
    
    if multi_index_df and index_columns:
      fa_transformed = fa.fit_transform(input_df)
      factor_scores_df = build_multi_index_dataframe(
        data=fa_transformed, 
        multi_index_df=multi_index_df, 
        index_columns=index_columns,
        columns=factor_labels)
      report.factor_scores = factor_scores_df
      
    if summary_plot is not None:
      from .plots import plot_correlation_heatmap
      # heatmap
      plot_correlation_heatmap(input_df, **kwargs)
    
    return loadings_df, report


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
  
  return ClusterReport(
    data=Z, 
    cophentic_corr=cophenetic_corr_coeff, 
    cluster_labels=cluster_labels)


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


def intra_cluster_variation(
  k: int, 
  data: ArrayLike,
  callback: ty.Callable[[int, ArrayLike, ty.Any], None] = None, 
  **kwargs) -> ArrayLike:
  wss = []
  for i in range(k):
    cluster = AgglomerativeClustering(
      n_clusters=i+1,
      affinity='euclidean',
      linkage='average')
    cluster.fit_predict(data)
    # cluster index or labels
    labels = cluster.labels_
    stderr.stderr.print(f'Cluster {i+1}: {labels}')
    ds = []
    for j in range(i+1):
      cluster_data = data[labels == j]
      cluster_mean = np.mean(cluster_data, axis=0)
      ds += [np.linalg.norm(cluster_data - cluster_mean, axis=1)]
    wss.append(np.sum(ds))
  
  if callback:
    # SEE arrays#simple_plot function
    callback(k, wss, **kwargs)

  return wss



if __name__ == "__main__":
  pass
