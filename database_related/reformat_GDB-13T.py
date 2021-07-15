from qm7b_t_format_specs import create_xyz_files
import os

original_data_dir=os.environ["DATA"]+"/GDB13-T"

formatted_data_dir=os.environ["DATA"]+"/GDB13T_reformatted"

create_xyz_files(original_data_dir, formatted_data_dir)
