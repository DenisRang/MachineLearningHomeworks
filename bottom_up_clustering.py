import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from matplotlib import markers
import itertools


# finds indices of closest clusters to be merged on next iteration
# clusters_matrix - clusters matrix
# distance_col, cluster_col - ids of columns keeping min distance and closest cluster id
# returns cluster indices and distance between them
def find_clusters_to_merge(clusters_matrix, distance_col, cluster_col):
    c1_id = np.argmin(clusters_matrix[:, distance_col])
    c2_id = clusters_matrix[c1_id, cluster_col]
    distance = clusters_matrix[c1_id, distance_col]
    return int(c1_id), int(c2_id), distance


# Transforms distances of a point in 'except_cluster' between other points
# into distances of the point between other clusters
# returns {cluster_1:[distance_1,distance_2], cluster_2:[distance_3]}, where
# distance_1 are distance between 'except_cluster' and cluster_1
# distance_2 are distance between 'except_cluster' and cluster_1
# distance_3 are distance between 'except_cluster' and cluster_2
def transform_to_cluster_distances(except_cluster, point_distances, X_matrix):
    result = {point: [] for point in np.unique(X_matrix[:, -1])}
    result.pop(except_cluster)
    num_points = len(point_distances)
    for i in range(num_points):
        cluster = X_matrix[i, -1]
        if (cluster != except_cluster):
            result[cluster].append(point_distances[i])
    return result


# performs merge of clusters with indices c1_index, c2_index
# updates single-linkage distances in clusters_matrix
# updates cluster membership column in X_matrix
# c1_index, c2_index - indices of clusters to be merged
# X_matrix - data + cluster membership column
# distance_col, cluster_col - ids of columns keeping min distance and closest cluster id
# distances_matrix - initial pairwise distances matrix, good implementation doesn't need it in this method
def single_link_merge(c1_index, c2_index, X_matrix, clusters_matrix, distance_col, cluster_col, distances_matrix):
    for i in range(X_matrix.shape[0]):
        if (X_matrix[i, -1] == c2_index):
            X_matrix[i, -1] = X_matrix[c1_index, -1]

    for i in range(clusters_matrix.shape[0]):
        cluster_distances = transform_to_cluster_distances(X_matrix[i, -1], clusters_matrix[i, :distance_col], X_matrix)
        if (len(cluster_distances) == 0):
            break
        for cluster in cluster_distances.keys():
            min_distance = min(cluster_distances[cluster])
            cluster_distances[cluster] = min_distance
        min_cluster_distance = min(cluster_distances.items(), key=lambda x: x[1])
        clusters_matrix[i, distance_col] = min_cluster_distance[1]
        clusters_matrix[i, cluster_col] = min_cluster_distance[0]
    pass


# performs merge of clusters with indices c1_index, c2_index
# updates complete-linkage distances in clusters_matrix
# updates cluster membership column in X_matrix
# c1_index, c2_index - indices of clusters to be merged
# X_matrix - data + cluster membership column
# distance_col, cluster_col - ids of columns keeping min distance and closest cluster id
# distances_matrix - initial pairwise distances matrix, good implementation doesn't need it in this method
def complete_link_merge(c1_index, c2_index, X_matrix, clusters_matrix, distance_col, cluster_col, distances_matrix):
    for i in range(X_matrix.shape[0]):
        if (X_matrix[i, -1] == c2_index):
            X_matrix[i, -1] = X_matrix[c1_index, -1]

    for i in range(clusters_matrix.shape[0]):
        cluster_distances = transform_to_cluster_distances(X_matrix[i, -1], clusters_matrix[i, :distance_col], X_matrix)
        if (len(cluster_distances) == 0):
            break
        for cluster in cluster_distances.keys():
            max_distance = max(cluster_distances[cluster])
            cluster_distances[cluster] = max_distance
        min_cluster_distance = min(cluster_distances.items(), key=lambda x: x[1])
        clusters_matrix[i, distance_col] = min_cluster_distance[1]
        clusters_matrix[i, cluster_col] = min_cluster_distance[0]
    pass


# performs merge of clusters with indices c1_index, c2_index
# updates average-linkage distances in clusters_matrix
# updates cluster membership column in X_matrix
# c1_index, c2_index - indices of clusters to be merged
# X_matrix - data + cluster membership column
# distance_col, cluster_col - ids of columns keeping min distance and closest cluster id
# distances_matrix - initial pairwise distances matrix, use it for this method
def average_link_merge(c1_index, c2_index, X_matrix, clusters_matrix, distance_col, cluster_col, distances_matrix):
    for i in range(X_matrix.shape[0]):
        if (X_matrix[i, -1] == c2_index):
            X_matrix[i, -1] = X_matrix[c1_index, -1]

    for i in range(clusters_matrix.shape[0]):
        cluster_distances = transform_to_cluster_distances(X_matrix[i, -1], clusters_matrix[i, :distance_col], X_matrix)
        if (len(cluster_distances) == 0):
            break
        for cluster in cluster_distances.keys():
            mean_distance = np.mean(cluster_distances[cluster])
            cluster_distances[cluster] = mean_distance
        min_cluster_distance = min(cluster_distances.items(), key=lambda x: x[1])
        clusters_matrix[i, distance_col] = min_cluster_distance[1]
        clusters_matrix[i, cluster_col] = min_cluster_distance[0]
    pass


# the function which performs bottom-up (agglomerative) clustering
# merge_func - one of the three merge functions above, each with different linkage function
# X_matrix - data itself
# threshold - maximum merge distance, we stop merging if we reached it. if None, merge until there only is one cluster
def bottom_up_clustering(merge_func, X_matrix, distances_matrix, threshold=None):
    num_points = X_matrix.shape[0]

    # take dataset, add and initialize column for cluster membership
    X_data = np.c_[X_matrix, np.arange(0, num_points, 1)]

    # create clusters matrix, initially consisting of all points and pairwise distances
    # with last columns being distance to closest cluster and id of that cluster
    clusters = np.c_[distances_matrix, np.zeros((num_points, 2))]

    # ids of added columns - column with minimal distances, column with closest cluster ids
    dist_col_id = num_points
    clust_col_id = num_points + 1

    # calculate closest clusters and corresponding distances for each cluster
    clusters[:, clust_col_id] = np.argmin(clusters[:, :num_points], axis=1)
    clusters[:, dist_col_id] = np.amin(clusters[:, :num_points], axis=1)

    # array for keeping distances between clusters that we are merging
    merge_distances = np.zeros(num_points - 1)
    # main loop. at each step we are identifying and merging two closest clusters (wrt linkage function)
    for i in range(0, num_points - 1):
        c1_id, c2_id, distance = find_clusters_to_merge(clusters, dist_col_id, clust_col_id)

        # if threshold is set, we don't merge any further if we reached the desired max distance for merging
        if threshold is not None and distance > threshold:
            break
        merge_distances[i] = distance
        merge_func(c1_id, c2_id, X_data, clusters, dist_col_id, clust_col_id, distances_matrix)

    # todo use the plot below to find the optimal threshold to stop merging clusters
    plt.plot(np.arange(0, num_points - 1, 1), merge_distances[:num_points - 1])
    plt.title("Merge distances over iterations")
    plt.xlabel("Iteration #")
    plt.ylabel("Distance")
    plt.show()

    for k, (marker, color) in zip(range(num_points), itertools.product(markers, colormap)):
        plt.scatter(X_data[X_data[:, 2] == k, 0], X_data[X_data[:, 2] == k, 1], color=color, marker=marker)
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.show()


# importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# creating and populating matrix for storing pairwise distances
# diagonal elements are filled with np.inf to ease further processing
distances = squareform(pdist(X, metric='euclidean'))
np.fill_diagonal(distances, np.inf)

# seting up colors and marker types to use for plotting
markers = markers.MarkerStyle.markers
colormap = plt.cm.Dark2.colors

# performing bottom-up clustering with three different linkage functions
# todo set your own thresholds for each method.
# todo find thresholds by looking at plot titled "Merge distances over iterations" when threshold is set to None
bottom_up_clustering(single_link_merge, X, distances, threshold=9)
bottom_up_clustering(complete_link_merge, X, distances, threshold=17)
bottom_up_clustering(average_link_merge, X, distances, threshold=15)
