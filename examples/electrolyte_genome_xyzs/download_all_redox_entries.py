from electrolyte_genome_format_specs import create_entry_xyz, QuantitiesNotAvailableError, get_results
import subprocess

dir_name='EGP_xyzs'

subprocess.run(["mkdir", '-p', dir_name])

quant_names=['oxidation_lithium', 'reduction_lithium']

quant_set=set(quant_names)

results=get_results({})
for mol_counter, molecule in enumerate(results):
    if quant_set.issubset(set(molecule.keys())):
        mol_id=molecule["task_id"]
        skip=False
#      First check the task_id is not repeated elsewhere.
        for mol_counter1, molecule1 in enumerate(results):
            if mol_counter != mol_counter1:
                if mol_id==molecule1["task_id"]:
                    skip=True
                    break
        if not skip:
            xyz_name=dir_name+"/"+mol_id+'.xyz'
            create_entry_xyz(xyz_name, mol_id, output_quants=quant_names)

