#!/usr/bin/env python

import typing as ty
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler

from .arrays import ArrayLike
from .console import new_progress_display, stderr
from .data import (_check_input_dataframe, build_multi_index_dataframe,
                   build_single_row_dataframe, get_records_match_condition,
                   normalize_columns)
from .highlights import highlight_eigenvalues
from .modules import install as install_package
from .plots import (find_no_clusters_by_dist_growth_acceleration_plot,
                    find_no_clusters_by_elbow_plot, make_dendrogram,
                    plot_column_correlation_heatmap, plot_factors_heatmap,
                    scree_plot)

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
class RolesReport:
  data: ArrayLike = field(default_factory=list)
  metrics: pd.DataFrame = None
  roles: ArrayLike = field(default_factory=list)
  
  def plot_parameters(self, **kwargs) -> None:
    k = find_no_clusters_by_dist_growth_acceleration_plot(self.data, **kwargs)
    output = kwargs.get('output', stderr)
    output.print(f"Number of clusters: {k}")


@dataclass(frozen=True)
class FactorAnalysisReport:
  metrics: pd.DataFrame = None
  factor_scores: pd.DataFrame = None
  factors: Styler = None


def factor_analysis(
  # FYI: this is the interest_df_norm
  input_df: pd.DataFrame,
  k: int = 10,
  metrics_only: bool = False,
  multi_index_df: pd.DataFrame = None,
  plot_summary: bool = False,
  rotation: str = None,
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
  report = replace(report, metrics=build_single_row_dataframe(metrics))
  
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
    
    no_factors = len(get_records_match_condition(eigenvalues_df, lambda x: x.Eigenvalue > 1.))
    metrics['Number_of_Factors'] = no_factors
    report = replace(report, metrics=build_single_row_dataframe(metrics))

    if plot_summary:
      # scree plot
      scree_plot(input_df.copy(), eigenvalues_df['Eigenvalue'].values.tolist(), **kwargs)
    # eigenvalues_df.style.apply(highlight_eigenvalues, color='yellow')
    
    report = replace(report, factors=eigenvalues_df.style.apply(highlight_eigenvalues, color='yellow'))
    
    return eigenvalues_df, report
  else:
    # Get factor loadings
    loadings_matrix = fa.loadings_
    factor_labels = ['Factor' + ' ' + str(i + 1) for i in range(k)]
    
    metrics['Number_of_Factors'] = k
    report = replace(report, metrics=build_single_row_dataframe(metrics))
    
    # factors' loadings
    loadings_df = pd.DataFrame(
      data = loadings_matrix, 
      index = input_df.columns, 
      columns = factor_labels)
        
    if multi_index_df is not None:
      fa_transformed = fa.fit_transform(input_df)
      factor_scores_df = build_multi_index_dataframe(
        data=fa_transformed, 
        multi_index_df=multi_index_df,
        columns=factor_labels)
      
      # Keep factor score between 0 and 1.
      factor_scores_df = normalize_columns(factor_scores_df)
      # capture the factor scores
      report = replace(report, factor_scores=factor_scores_df)
      
    if plot_summary:
      # heatmap
      heatmap_type: str = kwargs.get('heatmap_plot', 'factors_heatmap')
      if heatmap_type == 'factors_heatmap':
        plot_factors_heatmap(loadings_df, **kwargs)
      else:
        plot_column_correlation_heatmap(loadings_df, **kwargs)
    
    return loadings_df, report


def roles_discovery(input_df: pd.DataFrame, plot_summary: bool = True, **kwargs) -> ty.Tuple[pd.DataFrame, RolesReport]:  
  _check_input_dataframe(input_df)
  
  cached_Z = kwargs.get('data', None)
  Z = shc.linkage(input_df, method='ward') if cached_Z is None else cached_Z
  c, _ = cophenet(Z, pdist(input_df))
  cophenetic_corr_coeff = round(c, 2)
  
  if 'data' in kwargs:
    del kwargs['data']
  
  metrics = {'Cophenetic_Corr_Coeff' : cophenetic_corr_coeff}
  report = RolesReport(data=Z, metrics=build_single_row_dataframe(metrics))
  
  role_labels = []
  flat_option = kwargs.get('flat_option', False)
  if flat_option:
    no_clusters = kwargs.get('no_clusters', 5)
    criterion = kwargs.get('criterion', 'maxclust')
    role_labels = shc.fcluster(Z, no_clusters, criterion=criterion)
    report = replace(report, roles=role_labels)
  
  if plot_summary:
    make_dendrogram(Z, **kwargs)
  
  return input_df, report


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
  plot_summary: bool = True,
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
  
  if plot_summary:
    find_no_clusters_by_elbow_plot(k, wss, **kwargs)

  return wss



if __name__ == "__main__":
  pass
