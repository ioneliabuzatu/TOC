import config

from scenicsergio.scenic_sergio_flow import ScenicSergioFlow

flow = ScenicSergioFlow(
    filepath_control_adjacency=config.filepath_adjancies_control,
    filepath_disease_adjacency=config.filepath_adjancies_disease,
    filepath_to_save_interactions="scenicsergio/both_interactions.txt",
    filepath_to_save_regulons="scenicsergio/both_regulons.txt",
    select_percentile_adjacency=0.002
)
flow.make_grn_sergio(make_one_state_only=True, make_which_state="control")
