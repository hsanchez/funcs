#!/usr/bin/env python

import typing as ty
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler

from .arrays import ArrayLike, normalize_vector_norms
from .common import with_status
from .console import new_quiet_console, stderr
from .data import (_check_input_dataframe, build_multi_index_dataframe,
                   build_single_row_dataframe, get_records_match_condition,
                   normalize_columns)
from .highlights import highlight_eigenvalues
from .modules import import_module
from .modules import install as install_package
from .plots import (find_no_clusters_by_dist_growth_acceleration_plot,
                    find_no_clusters_by_elbow_plot, make_dendrogram,
                    plot_column_correlation_heatmap, plot_factors_heatmap,
                    plot_RCI_distribution, radar_plot, scree_plot)

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
  from sklearn.manifold import TSNE
except ImportError:
  install_package('scikit-learn')
  from sklearn.cluster import AgglomerativeClustering
  from sklearn.manifold import TSNE


@dataclass(frozen=True)
class RolesReport:
  data: ArrayLike = field(default_factory=list)
  metrics: pd.DataFrame = None
  roles: ArrayLike = field(default_factory=list)
  # redundancy to ensure we can plot it
  roles_data: pd.DataFrame = None
  
  def plot_parameters(self, output=None, **kwargs) -> None:
    if output is None:
      output = stderr
    if len(self.data) == 0:
      output.print("No data to plot")
      return

    k = find_no_clusters_by_dist_growth_acceleration_plot(self.data, **kwargs)
    output.print(f"Number of clusters: {k}")
  
  def plot_components(self) -> None:
    if self.roles_data is not None:
      radar_plot(self.roles_data)


@dataclass(frozen=True)
class FactorAnalysisReport:
  metrics: pd.DataFrame = None
  factor_scores: pd.DataFrame = None
  factors: Styler = None

@dataclass(frozen=True)
class Projection2D:
  # projection data
  Z: ArrayLike = None
  # neighbors
  N: ArrayLike = None
  # encoding for whether a point (the array index) 
  # is a center point (the bool value at that index)
  C: ArrayLike = None


def factor_analysis(
  # FYI: this is the interest_df_norm
  input_df: pd.DataFrame,
  k: int = 10,
  metrics_only: bool = False,
  multi_index_df: pd.DataFrame = None,
  plot_summary: bool = False,
  rotation: str = None, 
  quiet: bool = False,
  **kwargs) -> ty.Tuple[pd.DataFrame, FactorAnalysisReport]:
  
  _check_input_dataframe(input_df)
  
  the_console = stderr
  if quiet:
    the_console = new_quiet_console()

  # Metrics
  report = FactorAnalysisReport()
  
  @with_status(console=the_console, prefix='Compute metrics')
  def compute_metrics(df: pd.DataFrame) -> ty.Tuple[pd.DataFrame, dict]:
    chi_square_value, p_value = calculate_bartlett_sphericity(df)
    # kmo_all, kmo_model
    _, kmo_model = calculate_kmo(df)
    mts = {
      'Chi_Square_Value' : round(chi_square_value, 2),
      'P_Value' : p_value,
      'Kaiser_Meyer_Olkin_Score' : round(kmo_model, 2)}
    return build_single_row_dataframe(mts), mts

  metrics_df, metrics = compute_metrics(input_df)
  report = replace(report, metrics=metrics_df)
  
  if metrics_only:
    # Return untouched input_df 
    # and its factor analysis report
    return input_df, report
  
  # Create factor analysis object and perform factor analysis
  fa = FactorAnalyzer(k, rotation=rotation)
  fa.fit(input_df)
  
  @with_status(console=the_console, prefix='Compute eigenvalues')
  def compute_eigenvalues(df: pd.DataFrame, mts: dict, evs: ArrayLike, fr: FactorAnalysisReport) -> ty.Tuple[pd.DataFrame, FactorAnalysisReport]:
    factor_labels = ['Factor' + ' ' + str(i + 1) for i in range(len(df.columns))]
    evs_df = pd.DataFrame(data=evs, index=factor_labels, columns=["Eigenvalue"])
    
    no_factors = len(get_records_match_condition(evs_df, lambda x: x.Eigenvalue > 1.))
    mts['Number_of_Factors'] = no_factors
    updated_fr = replace(fr, metrics=build_single_row_dataframe(mts))
    return evs_df, updated_fr
  
  @with_status(console=the_console, prefix='Compute factor loadings')
  def compute_factor_loadings(
    f: FactorAnalyzer, n_facts: int,
    df: pd.DataFrame, idx_df: pd.DataFrame,
    mts: dict, fr: FactorAnalysisReport,
    ln_mats) -> ty.Tuple[pd.DataFrame, FactorAnalysisReport]:

    factor_labels = ['Factor' + ' ' + str(i + 1) for i in range(n_facts)]
    
    mts['Number_of_Factors'] = n_facts
    updated_fr = replace(fr, metrics=build_single_row_dataframe(mts))
    
    # factors' loadings
    lns_df = pd.DataFrame(
      data = ln_mats, 
      index = df.columns, 
      columns = factor_labels)
        
    if idx_df is not None:
      fa_transformed = f.fit_transform(df)
      factor_scores_df = build_multi_index_dataframe(
        data=fa_transformed, 
        multi_index_df=idx_df,
        columns=factor_labels)
      
      # Keep factor score between 0 and 1.
      factor_scores_df = normalize_columns(factor_scores_df)
      # capture the factor scores
      updated_fr = replace(updated_fr, factor_scores=factor_scores_df)
    
    return lns_df, updated_fr
  
  if rotation is None:
    # Check Eigenvalues
    ev, _ = fa.get_eigenvalues()
    eigenvalues_df, report = compute_eigenvalues(input_df, metrics, ev, report)    

    if plot_summary:
      # scree plot
      scree_plot(input_df.copy(), eigenvalues_df['Eigenvalue'].values.tolist(), **kwargs)
    
    report = replace(report, factors=eigenvalues_df.style.apply(highlight_eigenvalues, color='yellow'))
    
    return eigenvalues_df, report
  else:
    # Get factor loadings
    loadings_matrix = fa.loadings_
    loadings_df, report = compute_factor_loadings(fa, k, input_df, multi_index_df, metrics, report, loadings_matrix)

    if plot_summary:
      # heatmap
      heatmap_type: str = kwargs.get('heatmap_plot', 'factors_heatmap')
      if heatmap_type == 'factors_heatmap':
        plot_factors_heatmap(loadings_df, **kwargs)
      else:
        plot_column_correlation_heatmap(loadings_df, **kwargs)
    
    return loadings_df, report


def roles_discovery(input_df: pd.DataFrame, plot_summary: bool = True, quiet: bool = False, **kwargs) -> ty.Tuple[pd.DataFrame, RolesReport]:  
  _check_input_dataframe(input_df)
  
  the_console = stderr
  if quiet:
    the_console = new_quiet_console()

  cached_Z = kwargs.get('data', None)
  
  @with_status(console=the_console, prefix='Compute distance matrix')
  def compute_dist_matrix(df: pd.DataFrame, cached_Z: ArrayLike) -> ArrayLike:
    if cached_Z is None:
      Z_val = shc.linkage(df, method='ward')
    else:
      Z_val = cached_Z
    return Z_val
  
  Z = compute_dist_matrix(input_df, cached_Z)
  
  if 'data' in kwargs:
    del kwargs['data']
  
  @with_status(console=the_console, prefix='Compute metrics')
  def compute_metrics(df: pd.DataFrame, d: ArrayLike) -> pd.DataFrame:
    c, coph_dists = cophenet(d, pdist(df))
    metrics = {
      'Cophenetic_Correlation' : round(c, 2),
      'Cophenetic_Distance' : round(coph_dists.mean(), 2)}
    return build_single_row_dataframe(metrics)

  metrics_df = compute_metrics(input_df, Z)
  report = RolesReport(data=Z, metrics=metrics_df)
  
  @with_status(console=the_console, prefix='Compute roles')
  def compute_clusters(k: int, c: str, r: RolesReport, d: ArrayLike) -> RolesReport:
    role_labels = shc.fcluster(d, k, criterion=c)
    updated_report = replace(r, roles=role_labels)
    return updated_report

  flat_option = kwargs.get('flat_option', False)
  if flat_option:
    no_clusters = find_no_clusters_by_dist_growth_acceleration_plot(Z, quiet=True)
    criterion = kwargs.get('criterion', 'maxclust')
    report = compute_clusters(no_clusters, criterion, report, Z)
  
  if 'flat_option' in kwargs:
    del kwargs['flat_option']
    
  if 'criterion' in kwargs:
    del kwargs['criterion']
  
  if plot_summary:
    make_dendrogram(Z, **kwargs)
    
  if len(report.roles) == 0:
    return input_df, report
  
  @with_status(console=the_console, prefix='Process roles')
  def generate_roles_df(df: pd.DataFrame, rr: RolesReport) -> pd.DataFrame:
    # Add role labels to the df
    role_objects = []
    for r in np.unique(rr.roles):
      role_objects.append(df[rr.roles == r].mean(0))
    
    all_roles_df = pd.concat(role_objects, axis=1, ignore_index=True)
    return all_roles_df

  roles_df = generate_roles_df(input_df, report)
  report = replace(report, roles_data=roles_df)
  return roles_df, report


def compute_role_change_intensity(
  input_df: pd.DataFrame,
  roles_report: RolesReport,
  target_column: str = 'Role',
  single_rci: bool = True,
  plot_summary: bool = True,
  dist_fn: ty.Callable[..., float] = euclidean,
  quiet: bool = False, 
  **kwargs) -> ty.Union[float, pd.DataFrame]:
  _check_input_dataframe(input_df)
  
  the_console = stderr
  if quiet:
    the_console = new_quiet_console()
  
  roles_df = roles_report.roles_data
  roles = roles_report.roles
  
  @with_status(console=the_console, prefix='Process samples')
  def process_dataframe(df: pd.DataFrame, tc: str, rls: ArrayLike) -> pd.DataFrame:
    input_roles = df.copy()
    input_roles[tc] = rls
    
    roles_index = input_roles.reset_index()
    roles_index = roles_index.sort_values(by = 'sent_time')
    start_period = roles_index.sent_time.min()
    end_period = roles_index.sent_time.max()
    mask = lambda x: (x.sent_time > start_period) & (x.sent_time <= end_period)
    roles_index_sample = roles_index.loc[mask].dropna()
    roles_index_sample = roles_index_sample.set_index(['sent_time'])
    return roles_index_sample
  
  sorted_df = process_dataframe(input_df, target_column, roles)
  cluster_centers = {i + 1: roles_df[col].values.tolist()
                     for i, col in enumerate(roles_df.columns)}

  def compute_RCI(df: pd.DataFrame, tc: str, cc: ArrayLike) -> float:
    RCI = 0.0
    for (_,x),(_,y) in zip(df[:-1].iterrows(), df[1:].iterrows()):
      R_cur = y[tc].astype(np.int64)
      R_prev = x[tc].astype(np.int64)
      RCI += (dist_fn(cc[R_cur], cc[R_prev]))
    return round(np.log10(RCI), 2)
  
  @with_status(console=the_console, prefix='Compute total RCI')
  def compute_total_RCI(df: pd.DataFrame, tc: str, cc: ArrayLike) -> float:
    return compute_RCI(df, tc, cc)
  
  if single_rci:
    return compute_total_RCI(sorted_df, target_column, cluster_centers)
  
  # Otherwise, compute RCI for each sender_id
  @with_status(console=the_console, prefix='Compute RCI')
  def compute_RCI_for_each_sender(df: pd.DataFrame, tc: str, cc: ArrayLike) -> pd.DataFrame:
    grouped_samples = df.groupby('sender_id')
    role_changes_out = [(name, compute_RCI(group, tc, cc)) for name, group in grouped_samples]
    role_names = [n for (n, _) in role_changes_out]
    rci_vals = np.array([s for (_, s) in role_changes_out])
    rci_vals[rci_vals == -np.inf] = 0
    return pd.DataFrame(data=rci_vals, index=role_names, columns=["RCI"])
  
  RCI_df = compute_RCI_for_each_sender(sorted_df, target_column, cluster_centers)
  if plot_summary:
    plot_RCI_distribution(RCI_df, **kwargs)
  return RCI_df


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


def get_vector_norms(
  target_activities: ArrayLike, 
  timeline_slice_aligned_models: list, 
  timeline_slices: ArrayLike, 
  act2idx: dict) -> ArrayLike:
  
  # NOTE: timeline_slice_aligned_models is
  # [dynamic_model.embeddings[t] for t in range(len(dynamic_model.embeddings))]
  
  aligned_activity_norms = []
  for act in target_activities:
    # norms = []
    embeddings = []
    for time_slice in timeline_slices:
      act_emb_at_week = timeline_slice_aligned_models[timeline_slices.index(time_slice)][act2idx[act], :]
      embeddings.append(act_emb_at_week)
      # norm = np.linalg.norm(act_emb_at_week)
      # norms.append(norm)
    
    # norms = np.array(norms)
    # norms = norms / sum(norms)
    aligned_activity_norms.append(normalize_vector_norms(embeddings))
  return aligned_activity_norms



def build_tsne_projection(
  activity: str,
  time_slices: ArrayLike,
  trained_model: ty.Any,
  act2idx: dict,
  idx2act: dict,
  n_neighbors: int = 10,
  n_components: int = 2) -> ty.Dict[str, Projection2D]:
  
  X_input = []
  neighbors = []
  center_points = []
  
  if isinstance(time_slices, np.ndarray):
    time_slices = time_slices.tolist()
  
  for time_slice in time_slices:
    embedding = trained_model.embeddings[time_slices.index(time_slice)]
    embedding_norm = np.reshape(np.sqrt(np.sum(embedding**2, 1)),(embedding.shape[0], 1))
    embedding_normalized = np.divide(embedding, np.tile(embedding_norm, (1, embedding.shape[1])))

    v = embedding_normalized[act2idx[activity], :]
    d = np.dot(embedding_normalized, v)
    
    idx = np.argsort(d)[::-1]
    nearby_acts = [(idx2act[k], time_slice) for k in list(idx[:n_neighbors])]
    neighbors.extend(nearby_acts)
    
    for k in range(n_neighbors):
      center_points.append(k == 0)
    
    X_input.append(embedding[idx[:n_neighbors], :])
  
  X_input = np.vstack(X_input)
  tsne_model = TSNE(n_components=n_components, metric='euclidean', init='pca')
  Z_input = tsne_model.fit_transform(X_input)
  
  if isinstance(neighbors, np.ndarray):
    neighbors = neighbors.tolist()
  
  projection2D = Projection2D(Z=Z_input, N=neighbors, C=center_points)
  return dict({activity: projection2D})


def build_tsne_projections(
  activities: ty.List[str],
  time_slices: ArrayLike,
  trained_model: ty.Any,
  act2idx: dict,
  idx2act: dict,
  n_neighbors: int = 10,
  n_components: int = 2) -> ty.Dict[str, Projection2D]:
  
  projections = {}
  for activity in activities:
    projections.update(build_tsne_projection(
      activity,
      time_slices,
      trained_model,
      act2idx,
      idx2act,
      n_neighbors,
      n_components))
  
  return projections


def _get_distance_triplets(
  z_data: ArrayLike, 
  c_data: ArrayLike, 
  # ~50 weeks in a year
  scale: float = 50.) -> ty.Tuple[ArrayLike, ArrayLike, ArrayLike]:
  zp = z_data * 1.0
  zp[:,0] = zp[:,0] * 2.
  
  dist_matrix = np.zeros((z_data.shape[0], z_data.shape[0]))
  for k in range(z_data.shape[0]):
    dist_matrix[:, k] = np.sum((zp - np.tile(zp[k,:], (z_data.shape[0], 1)))**2., axis=1)
  dist_to_centers = dist_matrix[:, c_data]
  dist_to_centers = np.min(dist_to_centers, axis=1)
  
  dist_to_others = dist_matrix + np.eye(z_data.shape[0]) * scale
  dist_to_others_idx = np.argsort(dist_to_others, axis=1)
  dist_to_others = np.sort(dist_to_others, axis=1)
  
  return dist_to_centers, dist_to_others, dist_to_others_idx


def tracking_moving_activity(
  activity: str, 
  projections: ty.Dict[str, Projection2D],
  activities: ty.Set[str],
  context_activities: ty.Set[str],
  act2abbr: ty.Dict[str, str],
  trained_model: ty.Any,
  time_slices: ArrayLike,
  draw_trace: bool = True,
  k_nearest_neighbors: int = 10,
  filter_ctx_activities: bool = False,
  **kwargs) -> ArrayLike:
  from queue import Queue
  
  x_shift = kwargs.get('x_shift', 2)
  fontsize = kwargs.get('fontsize', 12)
  s = kwargs.get('s', 150)
  color = kwargs.get('color', 'none')
  edgecolor = kwargs.get('edgecolor', 'red')
  
  plt_module = kwargs.get('pyplot_module', None)
  if plt_module is None:
    plt = import_module('matplotlib.pyplot', 'matplotlib')
  else:
    plt = plt_module
  
  if 'x_shift' in kwargs:
    del kwargs['x_shift']
  
  if isinstance(time_slices, np.ndarray):
    time_slices = time_slices.tolist()
  
  def get_marker_options(act: str, club_a: ty.Set[str], club_c: ty.Set[str]) -> ty.Tuple[str, float, bool]:
    is_ctx_act = True in [act in c for c in club_c]
    is_rel_act = True in [act in c for c in club_a]
    alpha = None
    # alpha is none for ctx activities
    if is_ctx_act:
      marker = 'go'
    elif is_rel_act:
      marker = 'mo'
      alpha = 0.5
    else:
      # irrelevant activity
      marker = 'b.'
      alpha = 0.1
    return marker, alpha, is_ctx_act

  trajectory = []
  
  proj2d = projections[activity]
  d2c, _, _ = _get_distance_triplets(proj2d.Z, proj2d.C)

  # draw activity movement in time
  scope = Queue()
  for k in range(len(proj2d.N) - 1, -1, -1):
    if proj2d.C[k]:
      trajectory += [proj2d.Z[k,:]]
      # catch time period (e.g., week)
      scope.put(proj2d.N[k][1])

      if draw_trace:
        marker, _, is_ctx_act = get_marker_options(
          proj2d.N[k][0],
          activities, 
          context_activities)
        
        label = f'{act2abbr[proj2d.N[k][0]]}-{proj2d.N[k][1]}'
        y_shift = np.random.randn() * .2
        plt.plot(proj2d.Z[k,0], proj2d.Z[k,1], marker)
        plt.text(
          proj2d.Z[k,0] - x_shift, proj2d.Z[k,1] + y_shift, label, fontsize=fontsize)

        if is_ctx_act:
          plt.scatter(
            proj2d.Z[k,0], proj2d.Z[k,1], s=s, color=color, edgecolor=edgecolor)
  
  trajectory = np.vstack(trajectory)
  if not draw_trace:
    return trajectory
  
  # captures ctx activities nearby the current
  # activity point in time
  dists = []
  ctx_acts = []
  while not scope.empty():
    time_slice = scope.get()
    time_slice_idx = time_slices.index(time_slice)
    k_neighbors = trained_model.k_nearest(
      activity, k=k_nearest_neighbors)[time_slice_idx]
    
    if filter_ctx_activities:
      mu = np.mean([float(n[1]) for n in k_neighbors])
      k_neighbors = np.array([x for x in k_neighbors if float(x[1]) > mu])
    else:
      k_neighbors = np.array([x for x in k_neighbors])
    
    for w in [(nb[0], time_slice) for nb in k_neighbors]:
      if w not in proj2d.N: continue ;
      k = proj2d.N.index(w)
      dists += [d2c[k]]#/100]
      ctx_acts += [k]
  
  # draw points that represent ctx activities
  # nearby the current activity point in time
  dist_mu = np.mean(dists)
  for k in ctx_acts:
    if d2c[k] > dist_mu:
      continue
    # label = f' {config.act2abbr[N[k][0]]}-{N[k][1]}'
    label = f'{act2abbr[proj2d.N[k][0]]}-{proj2d.N[k][1]}'
    marker, alpha, _ = get_marker_options(
      proj2d.N[k][0],
      activities, 
      context_activities)
    if alpha is None:
      x_shift = 0.
    y_shift = np.random.randn() * .2
    if alpha is None:
      plt.plot(proj2d.Z[k,0] - x_shift, proj2d.Z[k,1], marker)
      plt.text(proj2d.Z[k,0] - x_shift, proj2d.Z[k,1] + y_shift, label, fontsize=fontsize)
      plt.scatter(proj2d.Z[k,0], proj2d.Z[k,1], s=s, color=color, edgecolor=edgecolor)
    else:
      plt.plot(proj2d.Z[k,0] - x_shift, proj2d.Z[k,1], marker, alpha=alpha)
      plt.text(
        proj2d.Z[k,0] - x_shift, proj2d.Z[k,1] + y_shift, label, fontsize=fontsize, alpha=alpha)
  
  plt.axis('off')
  plt.plot(trajectory[:,0], trajectory[:,1])
  plt.show()
  
  # After plotting the trajectory, return it to the caller.
  return trajectory
    

if __name__ == "__main__":
  pass
