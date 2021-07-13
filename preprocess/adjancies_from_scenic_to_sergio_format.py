""" Adjacencies columns are in order of {TF Target Importance} """
import numpy as np


def create_txt_file_for_sergio(adjacencies: np.ndarray, filepath_grn_for_sergio: str):
    tfs_set = np.unique(adjacencies[:, 0].flatten())
    target_genes_set = np.unique(adjacencies[:, 1].flatten())
    print(f"Found {len(tfs_set)} unique TFs and {len(target_genes_set)} unique target genes.")
    target_genes_ids = {k: v for v, k in enumerate(target_genes_set)}
    tf_ids = {k: v for v, k in enumerate(tfs_set)}

    target_genes_dict = make_target_genes_dict(adjacencies)

    parse_dict_target_tf(target_genes_dict, filepath_grn_for_sergio, target_genes_ids, tf_ids)

    print("Done formatting pySCENIC GRN to SERGIO format.")


def make_target_genes_dict(adjacencies: np.ndarray):
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


def parse_dict_target_tf(target_genes_dict: dict, filepath_grn_for_sergio: str, target_genes_ids: dict, tf_ids: dict):
    with open(filepath_grn_for_sergio, "w") as file:
        for target_gene_name, (tfs, importances) in target_genes_dict.items():
            assert len(tfs) == len(importances), "Error in creating the target genes dictionary!"
            target_gene_id = target_genes_ids[target_gene_name]
            tf_for_target_gene_ids = ",".join([str(float(tf_ids[tf])) for tf in tfs])
            importances_to_str = ",".join([str(importance) for importance in importances])
            file.write(f"{float(target_gene_id)},{float(len(tfs))},{tf_for_target_gene_ids},{importances_to_str}\n")


if __name__ == "__main__":
    adjacencies = np.load("../data/scenic/mouse/GSE133382.adjacencies.npy", allow_pickle=True)
    create_txt_file_for_sergio(adjacencies, filepath_grn_for_sergio="../data/scenic/mouse/interaction_cID.txt")
