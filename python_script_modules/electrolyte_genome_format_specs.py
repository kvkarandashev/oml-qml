from qml.oml_compound import OML_Slater_pair
import json
import os
import sys
if sys.version_info[0] == 2:
    from urllib import quote_plus
else:
    from urllib.parse import quote_plus

from qml.oml_compound import OML_compound

def_float_format='{:.8E}'

vacuum_to_lithium=1.4
au_to_eV_mult=27.2113961

def reduction_lithium(xyz_name, calc_type="UHF", basis="min_bas", use_Huckel=False, optimize_geometry=True):
    return Electron_Affinity(xyz_name, calc_type=calc_type, basis=basis, use_Huckel=use_Huckel, optimize_geometry=optimize_geometry)-vacuum_to_lithium

def oxidation_lithium(xyz_name, calc_type="UHF", basis="min_bas", use_Huckel=False, optimize_geometry=True):
    return Ionization_Energy(xyz_name, calc_type=calc_type, basis=basis, use_Huckel=use_Huckel, optimize_geometry=optimize_geometry)-vacuum_to_lithium

def Ionization_Energy(xyz_name, calc_type="HF", basis="min_bas", use_Huckel=False, optimize_geometry=True):
    return electron_energy_change(xyz_name, 1, calc_type=calc_type, basis=basis, use_Huckel=use_Huckel, optimize_geometry=optimize_geometry)

def Electron_Affinity(xyz_name, calc_type="HF", basis="min_bas", use_Huckel=False, optimize_geometry=True):
    return -electron_energy_change(xyz_name, -1, calc_type=calc_type, basis=basis, use_Huckel=use_Huckel, optimize_geometry=optimize_geometry)

def electron_energy_change(xyz_name, charge_change, calc_type="HF", basis="min_bas", use_Huckel=False, optimize_geometry=True):
    Slater_pair=OML_Slater_pair(xyz = xyz_name, mats_savefile = xyz_name, calc_type=calc_type,
        basis=basis, second_charge=charge_change, optimize_geometry=optimize_geometry, use_Huckel=use_Huckel)
    Slater_pair.run_calcs()
    return (Slater_pair.comps[1].e_tot-Slater_pair.comps[0].e_tot)*au_to_eV_mult

quant_properties = {'reduction_lithium' : ('V', reduction_lithium),
                    'oxidation_lithium' : ('V', oxidation_lithium)}

class Quantity:
    def __init__(self, quant_name):
        self.name=quant_name
        quant_tuple=quant_properties[quant_name]
        self.dimensionality=quant_tuple[0]
        self.estimate_function=quant_tuple[1]
    def extract_xyz(self, filename):
        file=open(filename, 'r')
        lines=file.readlines()
        output=None
        for l in lines:
            lsplit=l.split()
            if len(lsplit)>1:
                if lsplit[0] == self.name:
                    output=float(lsplit[2])
                    break
        file.close()
        return output
    def OML_calc_quant(self, xyz_name, calc_type="UHF", basis="min_bas", use_Huckel=False, optimize_geometry=True):
        return quant_properties[self.name][1](xyz_name, calc_type=calc_type, basis=basis, use_Huckel=use_Huckel, optimize_geometry=optimize_geometry)
    def write_byprod_result(self, val, io_out):
        io_out.write(str(self.qm9_id)+" "+str(val)+"\n")
        
def create_entry_xyz(xyz_name, mol_id, output_quants=None):
    xyz_coords=get_mol_xyz(mol_id)
    output=open(xyz_name, 'wb')
    output.write(xyz_coords)
    output.close()
    output=open(xyz_name, 'a')
    output.write('\n')
    molecule=get_mol_result(mol_id)
    if output_quants is None:
        output_quants=list(molecule.keys())
    for quant_name in output_quants:
        try:
            val=molecule[quant_name]
        except KeyError:
            print("Attempting to write a quantity not present in entry.")
            quit()
        try:
            val_string=def_float_format.format(val)
        except (TypeError, ValueError):
            val_string=str(val)
        output.write(quant_name+" : "+val_string+'\n')
    output.close()

def xyzs_with_available_quants(molecule_array, available_quants, cut_num_mols):
    for mol_id, molecule in enumerate(molecule_array):
        if (all([molecule.has_key(necessary_quant) for necessary_quant in available_quants])):
            create_entry_xyz(molecule, str(mol_id))
            
#   For importing from materials project website. Mostly borrowed from the tutorial.
import requests

def MAPI_KEY():
    try:
        return os.environ["MAPI_KEY"]
    except LookupError:
        print("MAPI_KEY environmental variable needs to be set.")
        quit()
        
urlpattern = {
    "results": "https://materialsproject.org/molecules/results?query={spec}",
    "mol_json": "https://materialsproject.org/molecules/{mol_id}/json",
    "mol_svg": "https://materialsproject.org/molecules/{mol_id}/svg",
    "mol_xyz": "https://materialsproject.org/molecules/{mol_id}/xyz",
}

def get_results(spec, fields=None):
    """Take a specification document (a `dict`), and return a list of matching molecules.
    """
    # Stringify `spec`, ensure the string uses double quotes, and percent-encode it...
    str_spec = quote_plus(str(spec).replace("'", '"'))
    # ...because the spec is the value of a "query" key in the final URL.
    url = urlpattern["results"].format(spec=str_spec)
    return (requests.get(url, headers={'X-API-KEY': MAPI_KEY()})).json()

def get_mol_result(mol_id):
    return get_results({"task_id": mol_id})[0]

def get_mol_xyz(mol_id):
    url = urlpattern["mol_xyz"].format(mol_id=mol_id)
    response = requests.get(url, headers={'X-API-KEY': MAPI_KEY()})
    return response.content



