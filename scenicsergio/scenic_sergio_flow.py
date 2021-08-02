import config
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
        self.check_matrices_shape()
        return self.control_adjancies, self.disease_adjancies

    def check_matrices_shape(self):
        assert self.control_adjancies.shape[0] == self.disease_adjancies.shape[0], print("Ouch!")
        if self.control_adjancies.shape[1] != self.disease_adjancies.shape[1]:
            raise NotImplementedError

    def make_grn_sergio(self):

        if self.select_percent_adjancies != 100:

            if self.select_percent_adjancies <= 0 or isinstance(self.select_percent_adjancies, float):
                raise EnvironmentError("'select_percent_adjancies' should be a int between [1,99]")

            self.control_adjancies, self.disease_adjancies = self.select_interactions_portion
            logging.info(f"The trimmed adjacencies matrix has shape: {self.control_adjancies.shape}")

        # tot_unique_ids, master_regulons = unique_ids_and_master_regulons(adjacencies)

        # genes_ids_to_names_mapping = {k: v for v, k in enumerate(tot_unique_ids)}
        # save_gene_ids_to_their_name_dict(genes_ids_to_names_mapping, save_txt_filepath)
        #
        # grouped_target_genes_dict = group_genes_to_transcription_factor_in_dict(adjacencies)
        #
        # parse_dict_target_tf(grouped_target_genes_dict, save_txt_filepath, genes_ids_to_names_mapping)
        print("Done formatting pySCENIC GRN to SERGIO format.")


    @property
    def select_interactions_portion(self):
        """Take only the target-TF interaction above this threshold. The original adjancies matrix is huge
        and this simplifies the initial implementation"""
        select_until_row_x = int(len(self.control_adjancies) * (self.select_percent_adjancies/100))
        return self.control_adjancies[:select_until_row_x], self.disease_adjancies[select_until_row_x]

    def unique_ids_and_master_regulons(self, adjacencies):
        tfs_set = np.unique(adjacencies[:, 0].flatten())
        target_genes_set = np.unique(adjacencies[:, 1].flatten())

        unique_master_regulons = set(tfs_set) - set(target_genes_set)
        tot_ids_with_replicates = np.zeros(len(tfs_set) + len(target_genes_set), dtype=object)
        tot_ids_with_replicates[:len(tfs_set)] = tfs_set
        tot_ids_with_replicates[len(tfs_set):] = target_genes_set

        tot_unique_ids = np.unique(tot_ids_with_replicates)

        print(f"Found unique TFs/genes {len(tot_unique_ids)}.")
        return tot_unique_ids, unique_master_regulons
