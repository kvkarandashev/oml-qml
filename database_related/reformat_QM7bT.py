from qm7b_t_format_specs import create_xyz_files
import os

original_data_dir=os.environ["DATA"]+"/QM7b-T/200"

formatted_data_dir=os.environ["DATA"]+"/QM7bT_reformatted"

create_xyz_files(original_data_dir, formatted_data_dir)
