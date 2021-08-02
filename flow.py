import config

from scenicsergio.scenic_sergio_flow import ScenicSergioFlow

flow = ScenicSergioFlow(
    filepath_control_adjancies=config.filepath_adjancies_control,
    filepath_disease_adjancies=config.filepath_adjancies_disease,
    filepath_to_save_interactions="scenicsergio/both_interactions.txt",
    filepath_to_save_regulons="scenicsergio/both_regulons.txt",
    select_percent_adjancies=2
)
flow.make_txt_sergio()
