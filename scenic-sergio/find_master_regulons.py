from time import time
import numpy as  np
import json
import matplotlib.pyplot as plt
import networkx as nx


def get_regulons_and_non_regulons(adjacencies):
    tfs = adjacencies[:, 0]
    targets = adjacencies[:, 1]
    regulons = np.setdiff1d(tfs, targets)
    non_regulons_pairs = [(tf, target) for tf, target in zip(tfs, targets) if tf not in regulons]
    return regulons, non_regulons_pairs


def draw_graph_to_find_regulons(adjacencies):
    """Based on https://stackoverflow.com/questions/20133479/how-to-draw-directed-graphs-using-networkx-in-python"""

    regulons_list, non_regulons_pairs = get_regulons_and_non_regulons(adjacencies)
    np.save("./master_regulons_array.npy", regulons_list)

    edges = [(i[0], i[1]) for i in adjacencies]

    G = nx.DiGraph()
    G.add_edges_from(edges)

    val_map = {master_regulon:np.random.rand() for master_regulon in regulons_list}

    values = [val_map.get(node, 0.25) for node in G.nodes()]

    # Specify the edges you want here
    red_edges = non_regulons_pairs
    black_edges = [edge for edge in G.edges() if edge not in red_edges]

    plt.figure(1, figsize=(12, 12))
    # Need to create a layout when doing separate calls to draw nodes and edges
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_color=values, node_size=500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='r', arrows=True)
    nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=True)
    plt.title("Gene Regulatory network-regulons have all the same color")
    plt.show()


if __name__ == "__main__":
    start_time = time()
    genes_names_to_ids = open("../data/scenic/mouse/interaction_cID_trimmed_by_importance.txt.json", "r").read()
    adjacencies = np.load("../data/scenic/mouse/GSE133382.adjacencies.trimmed.npy", allow_pickle=True)
    print(adjacencies.shape)
    draw_graph_to_find_regulons(adjacencies[:100])
    print(f"Time in sec: {time()-start_time:.3f}")
