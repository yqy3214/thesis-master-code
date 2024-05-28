from sklearn.metrics.pairwise import euclidean_distances, row_norms
from sklearn.utils.extmath import stable_cumsum
import numpy as np
import multiprocessing


def kmeans_plusplus_with_outliers(X, n_clusters, z, sample_weight, x_squared_norms=None, n_local_trials=None):
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters + 1, n_features), dtype=X.dtype)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly and track index of point
    center_id = np.random.randint(n_samples)
    indices = np.full(n_clusters + 1, -1, dtype=int)
    centers[0] = X[center_id]
    indices[0] = center_id

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms, squared=True
    )
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters + 1):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = np.random.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True
        )

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)

        # 初始化启发式去除outliers
        distance_to_candidates_copy = distance_to_candidates.copy()
        if z != 0:
            for i in range(n_local_trials):
                sorted_index = np.argsort(distance_to_candidates_copy[i])[::-1]
                tmp_weight = 0
                for j in sorted_index:
                    tmp_weight += sample_weight[j]
                    distance_to_candidates_copy[i][j] = 0
                    if tmp_weight >= z:
                        break

        candidates_pot = distance_to_candidates_copy.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        centers[c] = X[best_candidate]
        indices[c] = best_candidate

    return centers[:-1], indices[:-1]


def get_inertia_thread(X, sample_weight, centers, labels):
    inertia = 0
    for i in range(X.shape[0]):
        if labels[i] != -1:
            inertia += np.linalg.norm(X[i] - centers[labels[i]]) ** 2 * sample_weight[i]
    return inertia


def get_inertia(X, sample_weight, centers, labels, n_threads):
    size = X.shape[0] // n_threads
    t = [0] * (n_threads + 1)
    for i in range(1, n_threads + 1):
        t[i] = t[i - 1] + size + (1 if i <= X.shape[0] % n_threads else 0)
    pool = multiprocessing.Pool(n_threads)
    inertia = pool.starmap_async(get_inertia_thread, [(
        X[t[i]:t[i+1]], sample_weight[t[i]:t[i+1]], centers, labels[t[i]:t[i+1]]) for i in range(n_threads)]).get()
    pool.close()
    pool.join()
    return sum(inertia)


def lloyd_iter_with_outliers_thread(X, sample_weight, labels, k):
    centers = np.zeros((k, X.shape[1]))
    weight_in_clusters = np.zeros(k)
    for i in range(X.shape[0]):
        if labels[i] != -1:
            centers[labels[i]] += X[i] * sample_weight[i]
            weight_in_clusters[labels[i]] += sample_weight[i]
    return centers, weight_in_clusters


def lloyd_iter_with_outliers(X, sample_weight, z, x_squared_norms, centers, centers_new, weight_in_clusters, labels, center_shift, n_threads):
    # with out liers
    distance_to_centers = euclidean_distances(X, centers, X_norm_squared=x_squared_norms.reshape(-1, 1), squared=True)
    labels[:] = np.argmin(distance_to_centers, axis=1)
    distance_to_centers = np.min(distance_to_centers, axis=1)
    if z != 0:
        sorted_distance = distance_to_centers.argsort()[::-1]
        tmp_weight, total_weitht = 0, sum(sample_weight)
        for i in sorted_distance:
            labels[i] = -1
            tmp_weight += sample_weight[i]
            if tmp_weight >= z:
                break
        total_weitht -= tmp_weight

    if n_threads == 1:
        centers_new[:], weight_in_clusters[:] = lloyd_iter_with_outliers_thread(
            X, sample_weight, labels, centers.shape[0])
    else:
        size = X.shape[0] // n_threads
        t = [0] * (n_threads + 1)
        for i in range(1, n_threads + 1):
            t[i] = t[i - 1] + size + (1 if i <= X.shape[0] % n_threads else 0)
        pool = multiprocessing.Pool(n_threads)
        tmp_centers_thread = pool.starmap_async(lloyd_iter_with_outliers_thread, [(
            X[t[i]:t[i+1]], sample_weight[t[i]:t[i+1]], labels[t[i]:t[i+1]], centers.shape[0]) for i in range(n_threads)]).get()
        # tmp_centers_thread = pool.starmap_async(lloyd_iter_with_outliers_thread, [(1, 2, 3, 4) for i in range(n_threads)]).get()
        pool.close()
        pool.join()

        centers_new[:] = np.zeros(centers_new.shape)
        weight_in_clusters[:] = np.zeros(centers_new.shape[0])
        for tmp_centers, tmp_weight_in_clusters in tmp_centers_thread:
            centers_new += tmp_centers
            weight_in_clusters += tmp_weight_in_clusters

    for i in range(centers_new.shape[0]):
        if weight_in_clusters[i] != 0:
            centers_new[i] /= weight_in_clusters[i]
    center_shift[:] = np.sum((centers_new - centers) ** 2, axis=1)
    return


def kmeans_with_outliers_single_lloyd(X, sample_weight, centers_init, z, max_iter=300, x_squared_norms=None, tol=1e-4, n_threads=1):
    n_clusters = centers_init.shape[0]
    centers = centers_init
    centers_new = np.zeros_like(centers)
    labels = np.full(X.shape[0], -1, dtype=np.int32)
    labels_old = labels.copy()
    weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
    center_shift = np.zeros(n_clusters, dtype=X.dtype)
    strict_convergence = False

    for i in range(max_iter):
        print(i, end='\r')
        lloyd_iter_with_outliers(X, sample_weight, z, x_squared_norms, centers, centers_new,
                                 weight_in_clusters, labels, center_shift, n_threads)

        centers, centers_new = centers_new, centers

        if np.array_equal(labels, labels_old):
            # First check the labels for strict convergence.
            strict_convergence = True
            break
        else:
            # No strict convergence, check for tol based convergence.
            center_shift_tot = center_shift.sum()
            if i % 20 == 0:
                if center_shift_tot <= tol:
                    break
            if center_shift_tot <= tol:
                break

        labels_old[:] = labels

    if not strict_convergence:
        # rerun E-step so that predicted labels match cluster centers
        lloyd_iter_with_outliers(X, sample_weight, z, x_squared_norms, centers, centers,
                                 weight_in_clusters, labels, center_shift, n_threads)

    inertia = get_inertia(X, sample_weight, centers, labels, n_threads)

    return labels, inertia, centers, i + 1


def kmeans_with_outliers(X, n_clusters, z=0, sample_weight=None, n_init=5, max_iter=300, n_threads=1, input_centers_init=None):
    x_squared_norms = row_norms(X, squared=True)
    best_inertia, best_labels = None, None
    if sample_weight is None:
        sample_weight = np.ones(X.shape[0])
    if not (input_centers_init is None):
        n_init = 1
    for _ in range(n_init):
        if input_centers_init is None:
            # centers_init, _ = kmeans_plusplus_with_outliers(X, n_clusters, z, sample_weight, x_squared_norms)
            # centers_init, _ = kmeans_plusplus(X, n_clusters)
            centers_init = X[np.random.choice(X.shape[0], n_clusters, replace=False)]
        else:
            centers_init = input_centers_init
        # centers_init, _ = kmeans_plusplus(X, n_clusters, x_squared_norms=x_squared_norms)
        labels, inertia, centers, n_iter_ = kmeans_with_outliers_single_lloyd(
            X, sample_weight, centers_init, z, max_iter=max_iter, x_squared_norms=x_squared_norms, n_threads=n_threads)
        if best_inertia is None or inertia < best_inertia:
            best_labels = labels
            best_centers = centers
            best_inertia = inertia
            best_n_iter = n_iter_

    return best_labels, best_inertia, best_centers, best_n_iter