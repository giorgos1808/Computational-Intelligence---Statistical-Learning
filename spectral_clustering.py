# --Spectral Clustering--
# Tsakiris Giorgos

from keras.datasets import mnist, cifar10
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sn
from sklearn.decomposition import PCA
import time
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans


def spec_clustering(X, n_neighbors, n_clusters, num_eigens=20, eigen_gap=False, dataset=''):
    # Adjacency Matrix.
    adjacency_matrix_s = kneighbors_graph(X=X, n_neighbors=n_neighbors, mode='connectivity').toarray()

    # create the graph laplacian
    degree_matrix = np.diag(adjacency_matrix_s.sum(axis=1))
    graph_laplacian = degree_matrix - adjacency_matrix_s

    # find the eigenvalues and eigenvectors
    eigenvals, eigenvcts = np.linalg.eig(graph_laplacian)
    eigenvals, eigenvcts = np.real(eigenvals), np.real(eigenvcts)

    eigenvals_sorted_indices = np.argsort(eigenvals)
    eigenvals_sorted = eigenvals[eigenvals_sorted_indices]

    # project and transpose
    indices = eigenvals_sorted_indices[:n_clusters]
    proj_df = pd.DataFrame(eigenvcts[:, indices.squeeze()])
    proj_df.columns = ['v_' + str(c) for c in proj_df.columns]

    # plot eigenvals
    if eigen_gap:
        fig, ax = plt.subplots()
        sn.scatterplot(x=range(1, eigenvals_sorted_indices.size + 1), y=eigenvals_sorted, ax=ax)
        ax.set(title='Sorted Eigenvalues Graph Laplacian', xlabel='index', ylabel=r'$\lambda$')
        plt.savefig(str(dataset)+'_eigenvalue_graph_neighbors'+str(n_neighbors)+'_clusters'+str(n_clusters)+'_'+str(X.shape[0])+'.png')

        fig, ax = plt.subplots()
        sn.scatterplot(x=range(1, num_eigens + 1), y=eigenvals_sorted[:num_eigens], ax=ax)
        ax.set(title='Sorted Eigenvalues Graph Laplacian', xlabel='index 1-'+str(num_eigens), ylabel=r'$\lambda$')
        plt.savefig(str(dataset)+'_eigenvalue'+str(num_eigens)+'_graph_neighbors'+str(n_neighbors)+'_clusters'+str(n_clusters)+'_'+str(X.shape[0])+'.png')

        eigenvals_sorted_n = eigenvals_sorted[:num_eigens]
        max_gap = 0
        gap_pre_index = 0
        for i in range(1, eigenvals_sorted_n.size):
            gap = eigenvals_sorted_n[i] - eigenvals_sorted_n[i - 1]
            if gap > max_gap:
                max_gap = gap
                gap_pre_index = i - 1

        n_clusters = gap_pre_index + 1

    # K-means clustering
    k_means = KMeans(random_state=0, n_clusters=n_clusters)
    labels = k_means.fit_predict(proj_df)

    return labels, n_clusters


def run(dataset='mnist', max_ex=1000, pca_n_components = 50, n_dim=3, num_neighbors=30, num_clusters=10, num_eigens=20, eigen_gap=False):
    if dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset == 'cifar10':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    X = np.reshape(X, (X.shape[0], -1))
    y = np.reshape(y, (-1,))

    # normalize X -> [0,1] from [0,255]
    X = np.float32(X) / 255
    y = np.float32(y)

    X = X[:max_ex, :]
    y = y[:max_ex]

    with open('results_' + dataset + '_clustering.txt', 'a') as f:
        print('dataset: ' + dataset + ', pca: ' + str(pca_n_components) + ', tsne: ' + str(n_dim) + ', neighbors: ' + str(
            num_neighbors) + ', clusters: ' + str(num_clusters) + ', max_ex: ' + str(max_ex))
        f.write('dataset: ' + dataset + ', pca: ' + str(pca_n_components) + ', tsne: ' + str(n_dim) + ', neighbors: ' + str(
            num_neighbors) + ', clusters: ' + str(num_clusters) + ', max_ex: ' + str(max_ex) + '\n')

        # pca + tsne
        start_pca_tsne = time.time()
        pca_model = PCA(n_components=pca_n_components)
        pca_data = pca_model.fit_transform(X)
        tsne_model = TSNE(n_components=2, random_state=0)
        X_tsne = tsne_model.fit_transform(pca_data)
        end_pca_tsne = time.time()

        tsne_df = pd.DataFrame(data=X_tsne, columns=("Dim_1", "Dim_2"))
        sn.FacetGrid(tsne_df, height=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
        plt.savefig(dataset + '_pca' + str(pca_n_components) + '_tsne_without_labels' + str(max_ex) + '.png')

        tsne_data = np.vstack((X_tsne.T, y)).T
        tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))
        sn.FacetGrid(tsne_df, hue="label", height=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
        plt.savefig(dataset + '_pca' + str(pca_n_components) + '_tsne_with_labels' + str(max_ex) + '.png')
        print('PCA + tSNE time: %0.2f sec' % (end_pca_tsne - start_pca_tsne))
        f.write('PCA + tSNE time: %0.2f sec\n' % (end_pca_tsne - start_pca_tsne))

        # 2nd t-SNE
        start_pca_tsne = time.time()
        tsne_model_n = TSNE(n_components=n_dim, random_state=0, method='exact')
        X_tsne_n = tsne_model_n.fit_transform(X_tsne)
        end_pca_tsne = time.time()
        print('2nd tSNE time: %0.2f sec' % (end_pca_tsne - start_pca_tsne))
        f.write('2nd tSNE time: %0.2f sec\n' % (end_pca_tsne - start_pca_tsne))

        # Spectral Clustering from scratch
        start_spec_clustering = time.time()
        labels, num_clusters = spec_clustering(X_tsne_n, num_neighbors, num_clusters, num_eigens=num_eigens, eigen_gap=eigen_gap, dataset=dataset)
        end_spec_clustering = time.time()
        print('Spectral Clustering from scratch time: %0.2f sec' % (end_spec_clustering - start_spec_clustering))
        f.write('Spectral Clustering from scratch time: %0.2f sec\n' % (end_spec_clustering - start_spec_clustering))

        clustering_tsne_n_df = np.vstack((X_tsne.T, labels)).T
        clustering_tsne_n_df = pd.DataFrame(data=clustering_tsne_n_df, columns=("Dim_1", "Dim_2", "label"))
        sn.FacetGrid(clustering_tsne_n_df, hue="label", height=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
        plt.savefig(dataset + '_clustering' + '_pca' + str(pca_n_components) + '_dim' + str(n_dim) + '_neighbors' +
                    str(num_neighbors) + '_clusters' + str(num_clusters) + '_' + str(max_ex) + '.png')
        print('Silhouette Score(clusters=' + str(num_clusters) + '): ' + f'{silhouette_score(X_tsne, labels)}')
        f.write('Silhouette Score(clusters=' + str(num_clusters) + '): ' + f'{silhouette_score(X_tsne, labels)}\n')


if __name__ == '__main__':
    # pca + t-sne : pca_n_components

    # 2nd t-SNE : n_dim

    # Spectral Clustering :
    #   k neighbors graph : num_neighbors
    #   plot eigen : eigen_gap, num_eigens
    #   KMeans :  num_clusters

    run(dataset='mnist', max_ex=2000, pca_n_components=70, n_dim=4, num_neighbors=50, num_clusters=10, num_eigens=20, eigen_gap=False)
    run(dataset='cifar10', max_ex=2000, pca_n_components=30, n_dim=3, num_neighbors=50, num_clusters=10, num_eigens=20, eigen_gap=False)

