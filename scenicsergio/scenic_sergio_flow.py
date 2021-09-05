import json
import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

__functions_in_ScenicSergioFlow__ = {
    "self.open_adjacency_from_filepath_input", 
    "self.make_grn_sergio",
    "self.make_one_state_grn",
    "self.group_interactions",
    "self.add_interaction",
    "self.trim_interactions",
    "self.unique_ids_master_regulons_non_regulons_pairs",
}


class ScenicSergioFlow(object):
    def __init__(self,
                 filepath_to_save_interactions: str,
                 filepath_to_save_regulons: str,
                 select_percentile_adjacency=100,
                 filepath_control_adjacency: str = None,
                 filepath_disease_adjacency: str = None,
                 filepath_save_gene_name_to_id_mapping: str = None
                 ):
        """
        Output Adjacencies from scenic comes as: columns are in order of {TF \tab Target \tab Importance}
        select_percent_adjacency: the portion of the original matrix to work with
        """
        self.control_filepath = filepath_control_adjacency
        self.disease_filepath = filepath_disease_adjacency
        self.filepath_to_save_interactions = filepath_to_save_interactions
        self.filepath_to_save_regulons = filepath_to_save_regulons
        self.select_percent_adjacency = select_percentile_adjacency
        self.filepath_save_gene_name_to_id_mapping = filepath_save_gene_name_to_id_mapping
        self.control_adjacency = self.disease_adjacency = None

        self.possible_states_adjacency = {
            "control": self.control_adjacency,
            "disease": self.disease_adjacency,
            "both": (self.control_adjacency, self.disease_adjacency)
        }

        self.filepaths_states = {"control": self.control_filepath, "disease": self.disease_filepath}

    def open_adjacency_from_filepath_input(self, open_both=False, which_state_to_open=""):
        if open_both:
            self.control_adjacency = np.load(self.control_filepath, allow_pickle=True)
            self.disease_adjacency = np.load(self.disease_filepath, allow_pickle=True)
            return self.control_adjacency, self.disease_adjacency
        else:
            if not open_both and which_state_to_open in self.possible_states_adjacency.keys():
                self.possible_states_adjacency[which_state_to_open] = np.load(
                    self.filepaths_states[which_state_to_open], allow_pickle=True
                )
                return self.possible_states_adjacency[which_state_to_open]

    def make_grn_sergio(self, make_one_state_only=False, make_which_state="control"):
        if make_one_state_only and make_which_state in self.possible_states_adjacency.keys():
            self.possible_states_adjacency[make_which_state] = self.open_adjacency_from_filepath_input(
                open_both=False, which_state_to_open=make_which_state
            )
            self.make_one_state_grn(self.possible_states_adjacency[make_which_state])
        else:
            self.control_adjacency, self.disease_adjacency = self.open_adjacency_from_filepath_input(open_both=True)
            self.make_two_states_grn()

    def make_one_state_grn(self, adjacency):
        if self.select_percent_adjacency is not None:
            adjacency = self.trim_interactions(adjacency)

        cycle_edges_tobe_removed = self._break_cycles(adjacency)
        # logging.info("The interactions to be ignored/removed: ", cycle_edges_tobe_removed)
        tot_unique_ids, master_regulons, non_regulons_pairs = self.unique_ids_master_regulons_non_regulons_pairs(
            adjacency)
        genes_ids_to_names_mapping = {k: v for v, k in enumerate(tot_unique_ids)}
        self.save_gene_ids_to_their_name_dict(genes_ids_to_names_mapping)
        interactions_tfs_per_target_gene = self.format_sergio(
            adjacency,
            master_regulons,
            genes_ids_to_names_mapping,
            cycle_edges_tobe_removed,
        )
        # interactions_tfs_per_target_gene = self.group_interactions(
        #     adjacency,
        #     genes_ids_to_names_mapping,
        #     cycle_edges_tobe_removed
        # )
        logging.info("Saving the interactions to a txt file...")
        self.save_interactions_to_txt_file(interactions_tfs_per_target_gene)
        print("Done formatting pySCENIC GRN to SERGIO format.")

    def group_interactions(self, adjacency, genes_names_to_ids_mapping, interactions_to_ignore):
        """
        Each tf is added to the bounded gene in the dictionary.
        Example: {gene: ([tfs],[importance_values])}.
        Each interaction is a tuple of (tf, target_gene, importance).
        """
        interactions_tfs_per_target_gene = {}
        for interaction in adjacency:
            interactions_tfs_per_target_gene = self.add_interaction(
                interaction,
                interactions_to_ignore,
                genes_names_to_ids_mapping,
                interactions_tfs_per_target_gene
            )
        return interactions_tfs_per_target_gene

    @staticmethod
    def add_interaction(
            interaction,
            edges_to_ignore,
            genes_names_to_ids_mapping,
            interactions_tfs_per_target_gene
    ):
        tf, target_gene, importance = interaction
        if (tf, target_gene) in edges_to_ignore:
            print(f"Ignored this interaction: ({tf} -> {target_gene})!")
            return interactions_tfs_per_target_gene
        encoded_tf = genes_names_to_ids_mapping[tf]
        encoded_target_gene = genes_names_to_ids_mapping[target_gene]
        if encoded_target_gene not in interactions_tfs_per_target_gene:
            interactions_tfs_per_target_gene[encoded_target_gene] = ([encoded_tf], [importance])
        else:
            interactions_tfs_per_target_gene[encoded_target_gene][0].append(encoded_tf)
            interactions_tfs_per_target_gene[encoded_target_gene][1].append(importance)
        return interactions_tfs_per_target_gene

    def trim_interactions(self, adjacency):
        """Take only the target-TF interaction above this threshold. The original adjacency matrix is huge
        and this simplifies the initial implementation"""
        select_until_row_x = int(len(adjacency) * (self.select_percent_adjacency / 100))
        adjacency = adjacency[:select_until_row_x]
        logging.info(f"The trimmed adjacency has now #{select_until_row_x} interactions.")
        return adjacency

    def unique_ids_master_regulons_non_regulons_pairs(self, adjacency):
        tfs = adjacency[:, 0]
        targets = adjacency[:, 1]
        regulons = np.setdiff1d(tfs, targets)
        non_regulons_pairs = [(tf, target) for tf, target in zip(tfs, targets) if tf not in regulons]

        tot_ids_with_replicates = np.zeros(len(tfs) + len(targets), dtype=object)
        tot_ids_with_replicates[:len(tfs)] = tfs
        tot_ids_with_replicates[len(tfs):] = targets
        tot_unique_ids = np.unique(tot_ids_with_replicates)
        print(f"Found unique TFs/genes {len(tot_unique_ids)}.")

        return tot_unique_ids, regulons, non_regulons_pairs

    def save_gene_ids_to_their_name_dict(self, genes_ids_to_their_names_dict):
        with open(f"{self.filepath_save_gene_name_to_id_mapping}", "w") as file:
            json.dump(genes_ids_to_their_names_dict, file)

    def draw_graph(self, regulons_list, non_regulons_pairs, edges) -> []:
        """Based on https://stackoverflow.com/questions/20133479/how-to-draw-directed-graphs-using-networkx-in-python"""
        G = nx.DiGraph()
        G.add_edges_from(edges)

        val_map = {master_regulon: np.random.rand() for master_regulon in regulons_list}
        values = [val_map.get(node, 0.2) for node in G.nodes()]

        regulons_edges_red_edges = [edge for edge in G.edges() if edge not in non_regulons_pairs]

        plt.figure(1, figsize=(12, 12))
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_color=values, node_size=500)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, edgelist=regulons_edges_red_edges, edge_color='r', arrows=True)
        nx.draw_networkx_edges(G, pos, edgelist=non_regulons_pairs, arrows=True)
        plt.title("Gene Regulatory network - non master regulons have the same node color")

        plt.show()

        try:
            cycle_edge = nx.find_cycle(G)
            print(cycle_edge)
            return cycle_edge
        except nx.exception.NetworkXNoCycle:
            return []

    def _break_cycles(self, adjacency):
        break_cycles = False
        cycle_edges = []
        edges = [(i[0], i[1]) for i in adjacency]
        while not break_cycles:
            cycle_edges_ = self.check_if_graph_is_cycled(edges)
            cycle_edges += cycle_edges_
            edges = [(i[0], i[1]) for i in adjacency if (i[0], i[1]) not in cycle_edges]
            if not cycle_edges_:
                break_cycles = True
        return cycle_edges

    @staticmethod
    def check_if_graph_is_cycled(edges) -> []:
        G = nx.DiGraph()
        G.add_edges_from(edges)
        try:
            cycle_edge = nx.find_cycle(G)
            return cycle_edge
        except nx.exception.NetworkXNoCycle:
            return []

    def format_sergio(self, adjacency, regulons_list, genes_names_to_indices, edges_to_ignore):
        interactions_tfs_per_target_gene = {}
        visited_regulons = {k: False for k in regulons_list}

        with open(self.filepath_to_save_regulons, "w") as file_regulons:
            for interaction in adjacency:
                tf, target_gene, importance = interaction

                if tf in regulons_list and not visited_regulons[tf]:
                    file_regulons.write(f"{float(genes_names_to_indices[tf])},0.1,0.1\n")  # TODO replace 0.1
                    visited_regulons[tf] = True

                encoded_tf = genes_names_to_indices[tf]
                encoded_target_gene = genes_names_to_indices[target_gene]
                if encoded_target_gene not in interactions_tfs_per_target_gene:
                    interactions_tfs_per_target_gene[encoded_target_gene] = ([encoded_tf], [importance])
                else:
                    interactions_tfs_per_target_gene[encoded_target_gene][0].append(encoded_tf)
                    interactions_tfs_per_target_gene[encoded_target_gene][1].append(importance)

        return interactions_tfs_per_target_gene

    def save_interactions_to_txt_file(self, interactions_tfs_per_target_gene):
        with open(self.filepath_to_save_interactions, "w") as file:
            for target_gene_index, (tfs, importances) in interactions_tfs_per_target_gene.items():
                assert len(tfs) == len(importances), "Error in creating the target genes dictionary!"
                tf_for_target_gene_ids = ",".join([str(float(tf)) for tf in tfs])
                importance_values_to_str = ",".join([str(importance) for importance in importances])
                file.write(f"{float(target_gene_index)},{float(len(tfs))},{tf_for_target_gene_ids},"
                           f"{importance_values_to_str}\n")

    def save_master_regulons_to_txt_file(self, master_regulons_list):
        pass

    def make_two_states_grn(self):
        if self.select_percent_adjacency != 100:

            if self.select_percent_adjacency <= 0 or isinstance(self.select_percent_adjacency, float):
                raise EnvironmentError("'select_percent_adjacency' should be a int between [1,99]")

            self.trim_interactions(by_control_adjacency=True)
        tot_unique_ids_control, master_regulons_control = self.unique_ids_and_master_regulons(self.control_adjacency)
        tot_unique_ids_d, master_regulons_d = self.unique_ids_and_master_regulons(self.disease_adjacency)
        genes_names_to_ids_mapping = {k: v for v, k in enumerate(tot_unique_ids_control)}
        self.save_gene_ids_to_their_name_dict(genes_names_to_ids_mapping, "./scenicsergio/genes_ids_mapping.json")
        self.interactions_tfs_per_target_gene = {}
        visited_regulons = {k: False for k in tot_unique_ids_control}
        with open(self.filepath_to_save_regulons, "w") as file_regulons:
            for interaction in self.control_adjacency:
                tf, target_gene, importance_c = interaction

                match_in_disease = (self.disease_adjacency[:, 0] == tf) & (self.disease_adjacency[:, 1] == target_gene)
                found_interaction_in_disease_too, = np.where(match_in_disease)

                if tf in master_regulons_control and tf in master_regulons_d and not visited_regulons[tf]:
                    # if found_interaction_in_disease_too:
                    _, _, importance_d = self.disease_adjacency[found_interaction_in_disease_too].squeeze()
                    file_regulons.write(f"{float(genes_names_to_ids_mapping[tf])},{importance_c},{importance_d}\n")
                    visited_regulons[tf] = True
                    break

                if tf not in master_regulons_control and tf not in master_regulons_d:
                    if found_interaction_in_disease_too:
                        _, _, importance_d = self.disease_adjacency[found_interaction_in_disease_too].squeeze()

                        avg_importance = (importance_c + importance_d) / 2
                        self.add_interaction(genes_names_to_ids_mapping, avg_importance, target_gene, tf)
                    else:
                        self.add_interaction(genes_names_to_ids_mapping, importance_c, target_gene, tf)
                else:
                    logging.critical("Skipping the case when on gene is master in ony state only.")
        print("Done formatting pySCENIC GRN to SERGIO format.")
