# Procedures for creating xyz files from libe.json are part of the chemxpl repository since they rely on graph equivalency routines a lot for sorting the molecules.
from qml.oml_compound import OML_compound, OML_Slater_pair

class QuantityNotAvailableError(Exception):
    pass

def get_charge_spin(xyz_name):
    with open(xyz_name, 'r') as xyz_input:
        xyz_charge_spin_line=xyz_input.readlines()[1]
        for component in xyz_charge_spin_line.split():
            comp_split=component.split("=")
            if comp_split[0]=="charge":
                charge=int(comp_split[1])
            else:
                if comp_split[0]=="spin_multiplicity":
                    spin=int(comp_split[1])-1
    return charge, spin

def charge_spin_calc_type(xyz_file):
    charge, spin=get_charge_spin(xyz_file)
    if spin==0:
        calc_type="HF"
    else:
        calc_type="UHF"
    return charge, spin, calc_type


def potential_energy(xyz_name, **oml_calc_kwargs):
    charge, spin, calc_type=charge_spin_calc_type(xyz_name)
    oml_comp=OML_compound(xyz = xyz_name, mats_savefile = xyz_name, calc_type=calc_type, charge=charge, spin=spin, **oml_calc_kwargs)
    oml_comp.run_calcs()
    return oml_comp.e_tot

estimate_functions={"electronic_energy_Ha" : potential_energy}

class Quantity:
    def __init__(self, quant_name):
        self.name=quant_name
        self.est_func=estimate_functions[quant_name]
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

