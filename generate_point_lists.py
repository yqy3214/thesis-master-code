import math
import numpy as np

def generate_point_list(weights, n_features_dimention, center_list, z, a=None, b=None):
    if a is None:
        a = center_list.max()
    if b is None:
        b = 1 / math.sqrt(n_features_dimention)
    n_clusters = center_list.shape[0]
    n_samples = sum(weights) + z
    cluster_size = [0 for _ in range(n_clusters + 1)]
    for i in range(n_clusters):
        cluster_size[i + 1] = weights[i] + cluster_size[i]
    point_list = np.random.normal(0, b, size=(n_samples, n_features_dimention))
    for i in range(n_clusters):
        real_center = center_list[i]
        point_list[cluster_size[i]: cluster_size[i + 1]] += real_center.reshape(1, -1)
    point_list[cluster_size[i + 1]: n_samples] += np.random.uniform(low=-a, high=a, size=(z, n_features_dimention))
    return point_list


def generate_point_lists(point_lists_num, point_lists_size, n_features_dimention, barycenter_support_size, z=0, a=None, b=None):
    point_lists_size -= z
    if a is None:
        a = 10
    if b is None:
        b = 1 / math.sqrt(n_features_dimention)
    c = 1 * a
    center = np.random.uniform(low=-a, high=a, size=(barycenter_support_size, n_features_dimention))
    center_offset_list = np.random.normal(0, b, size=(point_lists_num, barycenter_support_size, n_features_dimention))
    weight_on_barycenter = np.random.randint(point_lists_size // 2 // barycenter_support_size, point_lists_size * 3 // 2 // barycenter_support_size + 2, size=barycenter_support_size)
    point_lists = []
    center_list = []
    for center_offset in center_offset_list:
        point_lists.append(generate_point_list(weight_on_barycenter, n_features_dimention, center_offset + center, z, c))
        center_list.append(center_offset + center)

    return point_lists, center, weight_on_barycenter, center_list