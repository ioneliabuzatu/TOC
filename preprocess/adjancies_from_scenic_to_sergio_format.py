""" Adjacencies columns are in order of {TF \tab Target \tab Importance} """
import json
import numpy as np


def create_txt_file_for_sergio_from_scenic(adjacencies: np.ndarray, save_txt_filepath: str, importance_threshold=None):
    if importance_threshold is not None:
        adjacencies = select_grn_by_importance_threshold(adjacencies, importance_threshold)
        print(f"The trimmed adjacencies matrix has shape: {adjacencies.shape}")
        np.save("../data/scenic/mouse/GSE133382.adjacencies.trimmed.npy", adjacencies)

    tot_unique_ids, master_regulons = unique_ids_and_master_regulons(adjacencies)

    genes_ids_to_names_mapping = {k: v for v, k in enumerate(tot_unique_ids)}
    save_gene_ids_to_their_name_dict(genes_ids_to_names_mapping, save_txt_filepath)

    grouped_target_genes_dict = group_genes_to_transcription_factor_in_dict(adjacencies)

    parse_dict_target_tf(grouped_target_genes_dict, save_txt_filepath, genes_ids_to_names_mapping)

    print("Done formatting pySCENIC GRN to SERGIO format.")


def select_grn_by_importance_threshold(adjacencies: np.ndarray, importance_threshold: float) -> np.ndarray:
    """Take only the target-TF interaction above this threshold. The original adjancies matrix is huge
    and this simplifies the initial implementation"""
    select_until_row_x = int(len(adjacencies) * importance_threshold)
    return adjacencies[:select_until_row_x]


def unique_ids_and_master_regulons(adjacencies) -> np.ndarray:
    tfs_set = np.unique(adjacencies[:, 0].flatten())
    target_genes_set = np.unique(adjacencies[:, 1].flatten())

    unique_master_regulons = set(tfs_set) - set(target_genes_set)
    tot_ids_with_replicates = np.zeros(len(tfs_set) + len(target_genes_set), dtype=object)
    tot_ids_with_replicates[:len(tfs_set)] = tfs_set
    tot_ids_with_replicates[len(tfs_set):] = target_genes_set

    tot_unique_ids = np.unique(tot_ids_with_replicates)

    print(f"Found unique TFs/genes {len(tot_unique_ids)}.")
    return tot_unique_ids, unique_master_regulons


def save_gene_ids_to_their_name_dict(genes_ids_to_their_names_dict, filepath_to_save):
    with open(f"{filepath_to_save}.json", "w") as file:
        json.dump(genes_ids_to_their_names_dict, file)


def group_genes_to_transcription_factor_in_dict(adjacencies: np.ndarray) -> dict:
    """Returns: {"transcription_factor_x":tuple([list of genes regulated],[list of importance values])}"""
    target_genes_dict = dict()
    for row_idx, tf_target_row in enumerate(range(adjacencies.shape[0])):
        transcription_factor_name = adjacencies[row_idx, 0]
        target_gene_name = adjacencies[row_idx, 1]
        importance_value = adjacencies[row_idx, 2]

        if target_gene_name not in target_genes_dict:
            target_genes_dict[target_gene_name] = tuple(([], []))

        target_genes_dict[target_gene_name][0].append(transcription_factor_name)
        target_genes_dict[target_gene_name][1].append(importance_value)

    return target_genes_dict


def parse_dict_target_tf(target_genes_dict: dict, filepath_grn_for_sergio: str, unique_genes_ids: dict):
    with open(filepath_grn_for_sergio, "w") as file:
        for target_gene_name, (tfs, importances) in target_genes_dict.items():
            assert len(tfs) == len(importances), "Error in creating the target genes dictionary!"
            target_gene_id = unique_genes_ids[target_gene_name]
            tf_for_target_gene_ids = ",".join([str(float(unique_genes_ids[tf])) for tf in tfs])
            importances_to_str = ",".join([str(importance) for importance in importances])
            file.write(f"{float(target_gene_id)},{float(len(tfs))},{tf_for_target_gene_ids},{importances_to_str}\n")
    print(f"Saved grn file to {filepath_grn_for_sergio}")


if __name__ == "__main__":
    adjacencies = np.load("../data/scenic/mouse/GSE133382.adjacencies.npy", allow_pickle=True)
    print(f"Original adjacencies matrix shape: {adjacencies.shape}")
    create_txt_file_for_sergio_from_scenic(adjacencies,
                                           save_txt_filepath="../data/scenic/mouse/interaction_cID_trimmed_by_importance.txt",
                                           importance_threshold=0.000001
                                           )
