import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import accuracy_score


def best_map(l1, l2):
    """
    Permute labels of l2 to match l1 as much as possible
    """
    if len(l1) != len(l2):
        print("L1.shape must == L2.shape")
        exit(0)

    label1 = np.unique(l1)
    n_class1 = len(label1)

    label2 = np.unique(l2)
    n_class2 = len(label2)

    n_class = max(n_class1, n_class2)
    G = np.zeros((n_class, n_class))

    for i in range(0, n_class1):
        for j in range(0, n_class2):
            ss = l1 == label1[i]
            tt = l2 == label2[j]
            G[i, j] = np.count_nonzero(ss & tt)

    # Find the best permutation using a custom algorithm (no linear_assignment)
    new_l2 = np.zeros(l2.shape, dtype=int)
    used = set()
    for i in range(n_class2):
        best_match = -1
        best_match_value = -1
        for j in range(n_class1):
            if j not in used:
                if G[j, i] > best_match_value:
                    best_match = j
                    best_match_value = G[j, i]
        used.add(best_match)
        new_l2[l2 == label2[i]] = label1[best_match]

    return new_l2


def evaluation(X_selected, n_clusters, y):
    # Create a KMeans clustering model
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300,
                    tol=0.0001, verbose=0,
                    random_state=None, copy_x=True, )

    # Fit the model on the selected features
    kmeans.fit(X_selected)

    # Get the predicted cluster labels
    y_predict = kmeans.labels_

    # Calculate Normalized Mutual Information (NMI)
    nmi = normalized_mutual_info_score(y, y_predict)

    # calculate ACC
    y_permuted_predict = best_map(y, y_predict)
    acc = accuracy_score(y, y_permuted_predict)

    return nmi, acc
