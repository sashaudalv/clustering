from sklearn.cluster import KMeans
import numpy

file_path = './number_pairs.txt'
clustered_pairs_file_path = './number_pairs_cluster_%i'


def load_pairs():
    pairs = []
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            pairs.append(tuple(map(float, line.split(' '))))

    return pairs


def verbose_clustering_matrix(matrix, n_clusters):
    print('index', 'cluster')
    for i, cluster in enumerate(matrix):
        print(i, '\t', cluster)

    count_clusters = [0 for i in range(n_clusters)]

    for cluster in matrix:
        count_clusters[cluster] += 1

    print("Pairs count", len(matrix))
    print("Number pairs of clusters:", count_clusters)


def write_clustered_pairs_to_files(pairs, matrix, n_clusters):
    clustered_pairs = [[] for i in range(n_clusters)]

    for i, cluster in enumerate(matrix):
        clustered_pairs[cluster].append(pairs[i])

    for i in range(n_clusters):
        with open(clustered_pairs_file_path % i, 'w+') as f:
            for pair in clustered_pairs[i]:
                f.write('%f %f\n' % (pair[0], pair[1]))


def main():
    pairs = load_pairs()
    numpy_pairs = numpy.array(pairs)
    kmeans = KMeans(n_clusters=2, random_state=0, verbose=True).fit(numpy_pairs)
    print('Clusters centers:', kmeans.cluster_centers_[0], kmeans.cluster_centers_[1])
    print('Clustering matrix:')
    print(kmeans.labels_)
    verbose_clustering_matrix(kmeans.labels_, kmeans.n_clusters)
    write_clustered_pairs_to_files(pairs, kmeans.labels_, kmeans.n_clusters)

main()
