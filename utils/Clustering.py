from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift

from utils import *


# --- public --- #


def cluster_refinement(class2values, algorithm, initialized=None, threshold=None, n_clusters=None):
    class2clusters = dict()
    for class_index, values in class2values.items():
        if initialized is not None and class_index in initialized:
            previous_clusters = initialized[class_index]
            initialized_class = previous_clusters.cluster_centers_
            n_clusters = previous_clusters.n_clusters
        else:
            initialized_class = None
            if n_clusters is not None:
                n_clusters = min(n_clusters, len(values))
        clusters = _cluster_refinement_class(values, algorithm=algorithm,
                                             initialized=initialized_class,
                                             threshold=threshold,
                                             n_clusters=n_clusters)
        print(" class", class_index, "will use", cluster_number(clusters), "clusters")
        class2clusters[class_index] = clusters
    return class2clusters


def cluster_number(clusterer):
    if isinstance(clusterer, KMeans):
        return clusterer.n_clusters
    elif isinstance(clusterer, MeanShift):
        return clusterer.cluster_centers_.shape[0]


def cluster_center(clusterer, cluster):
    return clusterer.cluster_centers_[cluster]


# --- private --- #


def _cluster_refinement_class(values, algorithm, initialized, threshold, n_clusters):
    if algorithm == "KMeans":
        return _cluster_refinement_class_kmeans(values, initialized, threshold, n_clusters)
    elif algorithm == "MeanShift":
        return _cluster_refinement_class_meanshift(values)
    else:
        raise(ValueError("Unknown algorithm: " + str(threshold)))


def _cluster_refinement_class_kmeans(values, initialized, threshold, n_clusters):
    if n_clusters is not None:
        return _cluster(values, initialized, n_clusters)

    n_clusters = 1
    n_values = len(values)
    assert n_values > 0
    clustered = _cluster(values, initialized, n_clusters)
    inertias = [clustered.inertia_]
    while n_values > n_clusters:
        n_clusters_new = n_clusters + 1
        clustered_new = _cluster(values, initialized, n_clusters_new)
        inertias.append(clustered_new.inertia_)

        if _terminate_clustering(inertias, threshold):
            break
        clustered = clustered_new
        n_clusters += 1
    return clustered


def _cluster_refinement_class_meanshift(values):
    clustered = MeanShift().fit(values)
    return clustered


def _cluster(values, initialized, n_clusters):
    if initialized is not None:
        clustered = KMeans(n_clusters, init=initialized).fit(values)
    else:
        clustered = KMeans(n_clusters).fit(values)
    # print("Number of clusters:", n_clusters, ", inertia:", clustered.inertia_)
    return clustered


def _terminate_clustering(inertias, threshold):
    # policy: compute relative improvement toward previous step
    assert len(inertias) > 1
    if inertias[-2] != 0:
        improvement = 1 - (inertias[-1] / inertias[-2])
    else:
        improvement = 0
    if isNaN(improvement):
        print("Warning: Got NaN in clustering evaluation. This is likely because there was no data.")
        return True
    return improvement < threshold
