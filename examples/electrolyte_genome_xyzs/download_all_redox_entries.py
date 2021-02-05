from electrolyte_genome_format_specs import create_entry_xyz, TaskIDRepeated, get_results
import subprocess

dir_name='EGP_xyzs'

subprocess.run(["mkdir", '-p', dir_name])

quant_names=['oxidation_lithium', 'reduction_lithium']

quant_set=set(quant_names)

results=get_results({})
for mol_counter, molecule in enumerate(results):
    if quant_set.issubset(set(molecule.keys())):
        mol_id=molecule["task_id"]
        xyz_name=dir_name+"/"+mol_id+'.xyz'
        try:
            create_entry_xyz(xyz_name, mol_id, output_quants=quant_names)
        except TaskIDRepeated:
            print("Duplicated task_id:", mol_id)
