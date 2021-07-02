from qml.oml_compound import OML_compound, OML_Slater_pair
import subprocess
import qcportal as ptl

class QuantityNotAvailableError(Exception):
    pass

au_to_kcalmol_mult=627.50960803

angstrom_to_au=1.8897259885789

#   For creating the xyz files from api.qcarchive.molssi.org
def replace_spaces(input_str):
    output_str=""
    for char in input_str:
        if char == " ":
            true_char="_"
        else:
            true_char=char
        output_str+=true_char
    return output_str

def quer_kwargs2name(quer_kwargs):
    name=""
    for kword in quer_kwargs:
        if name != "":
            name+="/"
        if kword != "native":
            name+=replace_spaces(quer_kwargs[kword])
    return name

def create_xyz_dir(xyz_dir_name, additional_info_file=None):
    subprocess.run(["mkdir", xyz_dir_name])

    client=ptl.FractalClient(address="api.qcarchive.molssi.org")

    qm7b = client.get_collection("dataset", "qm7b")

    if additional_info_file is not None:
        with open(additional_info_file, 'w') as add_info_output:
            print(qm7b.list_values(), file=add_info_output)
    qm7b_mols = qm7b.get_molecules()
    mols_index_vals=qm7b_mols.index.values

    quant_kwargs=[{"native" : False, "driver": "e1"}, {"native" : False, "driver" : "ea"}, {"native" : False, "driver" : "emax"}, {"native" : False, "driver" : "energy", "method" : "pbe0"},
                    {"native" : False, "driver" : "homo", "method" : "GW"},
                    {"native" : False, "driver" : "homo", "method" : "pbe0"},
                    {"native" : False, "driver" : "homo", "method" : "ZINDO/s"},
                    {"native" : False, "driver" : "lumo", "method" : "GW"},
                    {"native" : False, "driver" : "lumo", "method" : "pbe0"},
                    {"native" : False, "driver" : "lumo", "method" : "ZINDO/s"},
                    {"native" : False, "driver" : "imax"},
                    {"native" : False, "driver" : "ip"},
                    {"native" : False, "driver" : "polarizability", "method" : "Self-consistent screening"},
                    {"native" : False, "driver" : "polarizability", "method" : "pbe0"},
                    {"native" : True, "driver" : "energy", "method" : "b2plyp", "basis" : "aug-cc-pvdz"},
                    {"native" : True, "driver" : "energy", "method" : "b2plyp", "basis" : "aug-cc-pvtz"},
                    {"native" : True, "driver" : "energy", "method" : "b2plyp", "basis" : "def2-svp"},
                    {"native" : True, "driver" : "energy", "method" : "b2plyp", "basis" : "def2-tzvp"},
                    {"native" : True, "driver" : "energy", "method" : "b2plyp", "basis" : "sto-3g"},
                    {"native" : True, "driver" : "energy", "method" : "b3lyp", "basis" : "aug-cc-pvdz"},
                    {"native" : True, "driver" : "energy", "method" : "b3lyp", "basis" : "aug-cc-pvtz"},
                    {"native" : True, "driver" : "energy", "method" : "b3lyp", "basis" : "def2-svp"},
                    {"native" : True, "driver" : "energy", "method" : "b3lyp", "basis" : "def2-tzvp"},
                    {"native" : True, "driver" : "energy", "method" : "b3lyp", "basis" : "sto-3g"},
                    {"native" : True, "driver" : "energy", "method" : "wb97m-v", "basis" : "aug-cc-pvdz"},
                    {"native" : True, "driver" : "energy", "method" : "wb97m-v", "basis" : "aug-cc-pvtz"},
                    {"native" : True, "driver" : "energy", "method" : "wb97m-v", "basis" : "def2-svp"},
                    {"native" : True, "driver" : "energy", "method" : "wb97m-v", "basis" : "def2-tzvp"},
                    {"native" : True, "driver" : "energy", "method" : "wb97m-v", "basis" : "sto-3g"}
    ]
    quant_names=[]
    quant_arrs=[]
    for cur_quant_kwargs in quant_kwargs:
        quant_names.append(quer_kwargs2name(cur_quant_kwargs))
        cur_vals=qm7b.get_values(**cur_quant_kwargs)
        val_index_vals=cur_vals.index.values
        if not all(val_index_vals==mols_index_vals):
            quit("Something wrong: molecule-quantity index mismatch.")
        quant_arrs.append(cur_vals.values)

    for mol_id, mol_arr in enumerate(qm7b_mols.values):
        xyz_name="qm7b_"+str(mol_id+1).zfill(4)+".xyz"
        xyz_output=open(xyz_dir_name+"/"+xyz_name, 'w')
        mol=mol_arr[0]
        coords=mol.geometry/angstrom_to_au
        elements=mol.symbols
        xyz_output.write(str(len(elements))+"\n\n")
        for el, atom_coord in zip(elements, coords): 
            print(el, atom_coord[0], atom_coord[1], atom_coord[2], file=xyz_output)
        for quant_name, quant_arr in zip(quant_names, quant_arrs):
            print(quant_name, ":", quant_arr[mol_id][0], file=xyz_output)
        xyz_output.close()


#   For estimating quantities.
def ionization_potential(xyz_name, **kwargs):
    return electron_energy_change(xyz_name, 1, **kwargs)

def electron_affinity(xyz_name, **kwargs):
    return -electron_energy_change(xyz_name, -1, **kwargs)

def electron_energy_change(xyz_name, charge_change, first_calc_type="HF", second_calc_type="UHF", **oml_comp_kwargs):
    Slater_pair=OML_Slater_pair(xyz = xyz_name, mats_savefile = xyz_name, calc_type=first_calc_type, second_calc_type=second_calc_type,
        second_charge=charge_change, **oml_comp_kwargs)
    Slater_pair.run_calcs()
    return (Slater_pair.comps[1].e_tot-Slater_pair.comps[0].e_tot)*au_to_kcalmol_mult

#   Placeholder, probably need to come up with something better.
def excitation_energy(xyz_name, **kwargs):
    return (LUMO_en(xyz_name, **kwargs)-HOMO_en(xyz_name, **kwargs)) #*au_to_kcalmol_mult It seems like for some reason excitation energy is in a.u.

#   HOMO and LUMO have been copied from qm9_format_specs.py
def HOMO_en(xyz_name, **oml_kwargs):
    oml_comp=OML_compound(xyz = xyz_name, mats_savefile = xyz_name, **oml_kwargs)
    oml_comp.run_calcs()
    return oml_comp.HOMO_en()*au_to_kcalmol_mult

def LUMO_en(xyz_name, **oml_kwargs):
    oml_comp=OML_compound(xyz = xyz_name, mats_savefile = xyz_name, **oml_kwargs)
    oml_comp.run_calcs()
    return oml_comp.LUMO_en()*au_to_kcalmol_mult

def potential_energy(xyz_name, **oml_calc_kwargs):
    oml_comp=OML_compound(xyz = xyz_name, mats_savefile = xyz_name, **oml_calc_kwargs)
    oml_comp.run_calcs()
    return oml_comp.e_tot*au_to_kcalmol_mult

#   Placeholders for quantities I have no idea about.
def polarizability(xyz_name, **oml_kwargs):
    raise QuantityNotAvailableError

def max_absorption_intensity(xyz_name, **oml_kwargs):
    raise QuantityNotAvailableError

def excitation_energy_max_absorption(xyz_name, **oml_kwargs):
    raise QuantityNotAvailableError

#   For importing quantities.
estimate_functions={"e1" : excitation_energy, "ea" : electron_affinity, "emax" : excitation_energy_max_absorption, "energy" : potential_energy, "homo" : HOMO_en, "imax" : max_absorption_intensity, "ip" : ionization_potential, "lumo" : LUMO_en, "polarizability" : polarizability}


class Quantity:
    def __init__(self, quant_name):
        self.name=quant_name
        self.est_func=estimate_functions[quant_name.split('/')[0]]
    def extract_xyz(self, filename):
        file=open(filename, 'r')
        lines=file.readlines()
        output=None
        for l in lines:
            lsplit=l.split()
            if len(lsplit)==3:
                if lsplit[0] == self.name:
                    output=float(lsplit[2])
                    break
        file.close()
        if output is None:
            raise QuantityNotAvailableError
        else:
            return output
    def OML_calc_quant(self, xyz_name, **kwargs):
        return self.est_func(xyz_name, **kwargs)

