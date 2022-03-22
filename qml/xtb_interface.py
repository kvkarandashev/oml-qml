import numpy as np
from pyscf.tools import molden
import subprocess, os
from .utils import rmdir, mktmpdir, write_compound_to_xyz_file, write_bytes
from .aux_abinit_classes import Pseudo_MF

unknown_field_list=["[Title]"]

def delete_unknown_fields(molden_file_name, processed_molden_file_name, unknown_field_list_in):
    molden_file_input=open(molden_file_name, 'r')
    molden_file_lines=molden_file_input.readlines()
    molden_file_input.close()

    molden_file_output=open(processed_molden_file_name, 'w')
    for line in molden_file_lines:
        if (line[0]=="["):
            closing_bracket=line.find("]")
            if closing_bracket != -1:
                needs_writing=(line[:closing_bracket+1] not in unknown_field_list_in)
        if needs_writing:
            molden_file_output.write(line)
    molden_file_output.close()

def xtb_output_extract_e_tot(xtb_output):
    lines=xtb_output.split("\n")
    for l in lines:
        lsplit=l.split()
        if len(lsplit)==6:
            if ((lsplit[0]=="::") and (lsplit[1] == "total") and (lsplit[2] == "energy")):
                return float(lsplit[3])

def generate_pyscf_mf_mol(oml_compound, temp_dir_name=None):
    if temp_dir_name is None:
        workdir=mktmpdir()
    else:
        subprocess.run(["mkdir", "-p", temp_dir_name])
        workdir=temp_dir_name
    xyz_name="xtb_inp.xyz"
    os.chdir(workdir)
    write_compound_to_xyz_file(oml_compound, xyz_name)
    xtb_output=subprocess.run(["xtb", xyz_name, "--molden"], capture_output=True)
    e_tot=xtb_output_extract_e_tot(xtb_output.stdout.decode('utf-8'))

    if temp_dir_name is not None:
        # Save the stdout and stderr files.
        write_bytes(xtb_output.stdout, "xtb.stdout")
        write_bytes(xtb_output.stderr, "xtb.stderr")

    molden_init="molden.input"
    molden_new="molden_new.input"
    delete_unknown_fields(molden_init, molden_new, unknown_field_list)

    pyscf_mol, mo_energy, mo_coeff, mo_occ, syms, spins=molden.load(molden_new)
    mf_out=Pseudo_MF(e_tot=e_tot, mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ)
    os.chdir("..")
    if temp_dir_name is None:
        rmdir(workdir)
    return mf_out, pyscf_mol
