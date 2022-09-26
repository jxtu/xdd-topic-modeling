import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def scatter_viz(d2v_array: "np.ndarray", labels: "np.ndarray", out_fig_name, *, use_tne: bool):
    pca = PCA(n_components=50).fit_transform(d2v_array)
    pca = pca[:, :50]
    fig, ax = plt.subplots(figsize=(12, 12))
    if use_tne:
        reduced_embeddings = TSNE(n_components=2).fit_transform(pca)
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=10, c=labels)
    else:
        plt.scatter(pca[:, 0], pca[:, 1], s=10, c=labels)
    plt.savefig(out_fig_name)
