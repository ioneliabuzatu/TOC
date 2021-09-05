import config

from scenicsergio.scenic_sergio_flow import ScenicSergioFlow

flow = ScenicSergioFlow(
    filepath_control_adjacency=config.filepath_adjancies_control,
    filepath_disease_adjacency=config.filepath_adjancies_disease,
    filepath_to_save_interactions="scenicsergio/data/healthy/healthy_interactions.txt",
    filepath_to_save_regulons="scenicsergio/data/healthy/healthy_regulons.txt",
    filepath_save_gene_name_to_id_mapping="./scenicsergio/data/healthy/healthy_gene_names_mapping_to_ids_in_grn.json",
    select_percentile_adjacency=0.0001
)
flow.make_grn_sergio(make_one_state_only=True, make_which_state="control")
