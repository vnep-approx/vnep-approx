import numpy as np

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances, pairwise_distances
from sklearn.decomposition import PCA

from experimental_orientation import paper_algorithm
from experimental_label_optimization import get_bag_width
from experimental_multiroot import generate_random_request_graph


def generate_and_store_data():
    req = generate_random_request_graph(10, 0.2)

    edges_ordered = sorted(req.edges)
    data = []
    sizes = []
    for oriented in paper_algorithm(req):
        sizes.append(get_bag_width(oriented))
        data.append(tuple(compare_edges(oriented, ij) for ij in edges_ordered))

    data_as_array = np.array(data)
    sizes_as_array = np.array(sizes)
    similarities = pairwise_distances(data_as_array, metric="manhattan")

    data_as_array.dump("out/data/data.mat")
    sizes_as_array.dump("out/data/sizes.mat")
    similarities.dump("out/data/simil.mat")

def compare_edges(oriented, ij):
    if ij in oriented.edges:
        return 1
    else:
        return -1

def load_data():
    data_as_array = np.load("out/data/data.mat")
    sizes_as_array = np.load("out/data/sizes.mat")
    similarities = np.load("out/data/simil.mat")
    return data_as_array, sizes_as_array,similarities


def orientation_distance_eval():
    _, sizes_as_array, similarities = load_data()
    size_diff = [
        abs(sizes_as_array[i] - sizes_as_array[j])
        for i in range(len(sizes_as_array)) for j in range(len(sizes_as_array))
        if .5 * similarities[i, j] <= 1
    ]

    plt.hist(size_diff, bins=np.arange(max(size_diff)+1)-0.5)
    plt.show()

def mds_eval():
    seed = 0
    _, sizes_as_array, similarities = load_data()
    print "similarities", similarities.shape

    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-7, random_state=seed,
                       dissimilarity="precomputed", n_jobs=1, verbose=100)
    pos = mds.fit(similarities).embedding_
    print "MDS 1"
    fig = plt.figure(1)
    ax = plt.axes([0., 0., 1., 1.])

    s = 100
    plt.scatter(pos[:, 0], pos[:, 1], cmap=plt.cm.viridis_r, c=sizes_as_array, s=s, lw=0, label='MDS')
    # plt.scatter(npos[:, 0], npos[:, 1], cmap=plt.cm.Reds, c=sizes_as_array,s=s, lw=0, label='NMDS')
    plt.legend(scatterpoints=1, loc='best', shadow=False)

    # Plot the edges
    start_idx, end_idx = np.where(pos)
    segments = np.array([
        [pos[i, :], pos[j, :]]
        for i in range(len(pos)) for j in range(len(pos))
    ])
    index = [
        .5 * similarities[i, j] <= 1 and (sizes_as_array[i] == sizes_as_array.min())
        for i in range(len(pos)) for j in range(len(pos))
    ]
    values = np.abs(similarities)
    lc = LineCollection(segments[index],
                        zorder=0, cmap=plt.cm.Blues_r,
                        norm=plt.Normalize(0, values.flatten()[index].max() + 0.5))
    lc.set_array(similarities.flatten()[index])
    lc.set_linewidths(0.5 * np.ones(len(segments)))
    ax.add_collection(lc)
    plt.colorbar()
    plt.show()



if __name__ == "__main__":
    while True:
        generate_and_store_data()
        # mds_eval()
        orientation_distance_eval()
