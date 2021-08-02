from time import time
import numpy as np
import json
import matplotlib.pyplot as plt
import networkx as nx


def draw_graph_master_regulons(adjacencies, regulons_list, non_regulons_pairs):
    """Based on https://stackoverflow.com/questions/20133479/how-to-draw-directed-graphs-using-networkx-in-python"""

    edges = [(i[0], i[1]) for i in adjacencies]
    G = nx.DiGraph()
    G.add_edges_from(edges)

    val_map = {master_regulon: np.random.rand() for master_regulon in regulons_list}
    values = [val_map.get(node, 0.2) for node in G.nodes()]

    regulons_edges_red_edges = [edge for edge in G.edges() if edge not in non_regulons_pairs]

    plt.figure(1, figsize=(12, 12))
    # Need to create a layout when doing separate calls to draw nodes and edges
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_color=values, node_size=500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=regulons_edges_red_edges, edge_color='r', arrows=True)
    nx.draw_networkx_edges(G, pos, edgelist=non_regulons_pairs, arrows=True)
    plt.title("Gene Regulatory network - non master regulons have the same node color")
    plt.show()


def master_regulons_and_non_master_regulons(adjacencies):
    tfs = adjacencies[:, 0]
    targets = adjacencies[:, 1]
    regulons = np.setdiff1d(tfs, targets)
    non_regulons_pairs = [(tf, target) for tf, target in zip(tfs, targets) if tf not in regulons]
    return regulons, non_regulons_pairs


def save_regulons_to_txt(regulons_list):
    with open('./regulons.txt', 'w') as f:
        master_regulons_each_newline = map(lambda x: x + '\n', regulons_list)
        f.writelines(master_regulons_each_newline)


def format_sergio(adjacencies, regulons_list, genes_names_to_indices):
    interactions_tfs_per_target_gene = {}
    visited_regulons = []

    with open("./regulons.txt", "w") as file_regulons:
        for interaction in adjacencies:
            tf, target_gene, importance = interaction

            if tf in regulons_list and tf not in visited_regulons:
                file_regulons.write(f"{float(genes_names_to_indices[tf])},0.1\n")
                visited_regulons.append(tf)

            encoded_tf = genes_names_to_indices[tf]
            encoded_target_gene = genes_names_to_indices[target_gene]
            if encoded_target_gene not in interactions_tfs_per_target_gene:
                interactions_tfs_per_target_gene[encoded_target_gene] = ([encoded_tf], [importance])
            else:
                interactions_tfs_per_target_gene[encoded_target_gene][0].append(encoded_tf)
                interactions_tfs_per_target_gene[encoded_target_gene][1].append(importance)

    with open("./interactions.txt", "w") as file:
        for target_gene_index, (tfs, importances) in interactions_tfs_per_target_gene.items():
            assert len(tfs) == len(importances), "Error in creating the target genes dictionary!"
            tf_for_target_gene_ids = ",".join([str(float(tf)) for tf in tfs])
            importance_values_to_str = ",".join([str(importance) for importance in importances])
            file.write(f"{float(target_gene_index)},{float(len(tfs))},{tf_for_target_gene_ids},"
                       f"{importance_values_to_str}\n")


if __name__ == "__main__":
    start_time = time()
    with open("../data/scenic/mouse/interaction_cID_trimmed_by_importance.txt.json") as file:
        data = file.read()
        genes_names_to_ids = json.loads(data)

    adjacencies = np.load("../data/scenic/mouse/GSE133382.adjacencies.trimmed.npy", allow_pickle=True)
    print(f"Total TF-target pairs: {adjacencies.shape}")

    regulons_list, non_regulons_pairs = master_regulons_and_non_master_regulons(adjacencies)
    print(f"Regulons found: {len(regulons_list)} and non regulons found: {len(non_regulons_pairs)}")
    np.save("./master_regulons_array.npy", regulons_list)
    save_regulons_to_txt(regulons_list)
    draw_graph_master_regulons(adjacencies, regulons_list, non_regulons_pairs)

    format_sergio(adjacencies,regulons_list, genes_names_to_ids)

    print(f"Time in sec: {time() - start_time:.3f}")
