from electrolyte_genome_format_specs import create_entry_xyz, Quantity

mol_id='mol-39005'

xyz_name=mol_id+'.xyz.tmp' # Only adding the *.tmp in order git ignored it.

quant_names=['oxidation_lithium', 'reduction_lithium']

create_entry_xyz(xyz_name, mol_id, output_quants=quant_names)


for quant_name in quant_names:
    quantity=Quantity(quant_name)
    estimate=quantity.OML_calc_quant(xyz_name, basis='6-31g', optimize_geometry=False)
    print(quantity.name, ' estimate: ', estimate, ' true: ', quantity.extract_xyz(xyz_name))
