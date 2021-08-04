import config
import json
import tqdm
import pandas as pd
import logging
import numpy as np


class ScenicSergioFlow(object):
    def __init__(self,
                 filepath_control_adjancies: str,
                 filepath_disease_adjancies: str,
                 filepath_to_save_interactions: str,
                 filepath_to_save_regulons: str,
                 select_percent_adjancies: int = 100
                 ):
        """
        Output Adjacencies from scenic comes as: columns are in order of {TF \tab Target \tab Importance}
        select_percent_adjancies: the portion of the original matrix to work with
        """
        self.control_filepath = filepath_control_adjancies
        self.disease_filepath = filepath_disease_adjancies
        self.filepath_to_save_interactions = filepath_to_save_interactions
        self.filepath_to_save_regulons = filepath_to_save_regulons
        self.select_percent_adjancies = select_percent_adjancies
        self.control_adjancies, self.disease_adjancies = self.open_adjancies

    @property
    def open_adjancies(self):
        self.control_adjancies = np.load(self.control_filepath, allow_pickle=True)
        self.disease_adjancies = np.load(self.disease_filepath, allow_pickle=True)
        return self.control_adjancies, self.disease_adjancies

    def check_matrices_shape(self):
        pass

    def make_grn_sergio(self):

        if self.select_percent_adjancies != 100:

            if self.select_percent_adjancies <= 0 or isinstance(self.select_percent_adjancies, float):
                raise EnvironmentError("'select_percent_adjancies' should be a int between [1,99]")

            self.trim_interactions(by_control_adjancies=True)

        tot_unique_ids_control, master_regulons_control = self.unique_ids_and_master_regulons(self.control_adjancies)
        tot_unique_ids_d, master_regulons_d = self.unique_ids_and_master_regulons(self.disease_adjancies)

        genes_names_to_ids_mapping = {k: v for v, k in enumerate(tot_unique_ids_control)}
        self.save_gene_ids_to_their_name_dict(genes_names_to_ids_mapping, "./scenicsergio/genes_ids_mapping.json")

        self.interactions_tfs_per_target_gene = {}
        visited_regulons = {k: False for k in tot_unique_ids_control}
        with open(self.filepath_to_save_regulons, "w") as file_regulons:
            for interaction in self.control_adjancies:
                tf, target_gene, importance_c = interaction

                match_in_disease = (self.disease_adjancies[:, 0] == tf) & (self.disease_adjancies[:, 1] == target_gene)
                found_interaction_in_disease_too, = np.where(match_in_disease)

                if tf in master_regulons_control and tf in master_regulons_d and not visited_regulons[tf]:
                    # if found_interaction_in_disease_too:
                    _, _, importance_d = self.disease_adjancies[found_interaction_in_disease_too].squeeze()
                    file_regulons.write(f"{float(genes_names_to_ids_mapping[tf])},{importance_c},{importance_d}\n")
                    visited_regulons[tf] = True
                    break

                if tf not in master_regulons_control and tf not in master_regulons_d:
                    if found_interaction_in_disease_too:
                        _, _, importance_d = self.disease_adjancies[found_interaction_in_disease_too].squeeze()

                        avg_importance = (importance_c + importance_d) / 2
                        self.add_interaction(genes_names_to_ids_mapping, avg_importance, target_gene, tf)
                    else:
                        self.add_interaction(genes_names_to_ids_mapping, importance_c, target_gene, tf)
                else:
                    logging.critical("Skipping the case when on gene is master in ony state only.")

        print("Done formatting pySCENIC GRN to SERGIO format.")

    def add_interaction(self, genes_names_to_ids_mapping, importance, target, tf):
        encoded_tf = genes_names_to_ids_mapping[tf]
        encoded_target_gene = genes_names_to_ids_mapping[target]
        if encoded_target_gene not in self.interactions_tfs_per_target_gene:
            self.interactions_tfs_per_target_gene[encoded_target_gene] = ([encoded_tf], [importance])
        else:
            self.interactions_tfs_per_target_gene[encoded_target_gene][0].append(encoded_tf)
            self.interactions_tfs_per_target_gene[encoded_target_gene][1].append(importance)

    def trim_interactions(self, by_control_adjancies=True):
        """Take only the target-TF interaction above this threshold. The original adjancies matrix is huge
        and this simplifies the initial implementation"""
        if by_control_adjancies:
            select_until_row_x = int(len(self.control_adjancies) * (self.select_percent_adjancies / 100))
            self.control_adjancies = self.control_adjancies[:select_until_row_x]
            # else:
            select_until_row_x = int(len(self.disease_adjancies) * (self.select_percent_adjancies / 100))
            self.disease_adjancies = self.disease_adjancies[:select_until_row_x]
        logging.info(f"The trimmed adjacency has #{select_until_row_x} interactions.")

    def unique_ids_and_master_regulons(self, adjancies):
        tfs_set = np.unique(adjancies[:, 0].flatten())
        target_genes_set = np.unique(adjancies[:, 1].flatten())

        unique_master_regulons = set(tfs_set) - set(target_genes_set)
        tot_ids_with_replicates = np.zeros(len(tfs_set) + len(target_genes_set), dtype=object)
        tot_ids_with_replicates[:len(tfs_set)] = tfs_set
        tot_ids_with_replicates[len(tfs_set):] = target_genes_set

        tot_unique_ids = np.unique(tot_ids_with_replicates)

        print(f"Found unique TFs/genes {len(tot_unique_ids)}.")
        return tot_unique_ids, unique_master_regulons

    def save_gene_ids_to_their_name_dict(self, genes_ids_to_their_names_dict, filepath_to_save):
        with open(f"{filepath_to_save}", "w") as file:
            json.dump(genes_ids_to_their_names_dict, file)

    def format_sergio(self, adjacency, regulons_list, genes_names_to_indices):
        interactions_tfs_per_target_gene = {}
        visited_regulons = {k: False for k in regulons_list}

        with open("./regulons.txt", "w") as file_regulons:
            for interaction in adjacency:
                tf, target_gene, importance = interaction

                if tf in regulons_list and visited_regulons[tf]:
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

    def save_states_interactions(self):
        with open("./interactions.txt", "w") as file:
            for target_gene_index, (tfs, importances) in interactions_tfs_per_target_gene.items():
                assert len(tfs) == len(importances), "Error in creating the target genes dictionary!"
                tf_for_target_gene_ids = ",".join([str(float(tf)) for tf in tfs])
                importance_values_to_str = ",".join([str(importance) for importance in importances])
                file.write(f"{float(target_gene_index)},{float(len(tfs))},{tf_for_target_gene_ids},"
                           f"{importance_values_to_str}\n")
