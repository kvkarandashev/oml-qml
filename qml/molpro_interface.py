import numpy as np
from .oml_representations import AO
import subprocess, os
from .utils import rmdir, mktmpdir

def check_els(l1, l2):
    if (len(l1)<len(l2)):
        return False
    else:
        for lid, ell in enumerate(l2):
            if l1[lid]!=ell:
                return False
        return True

def mo_occs(noccs, naos):
    if len(noccs)==1:
        occupancy=2.0
    else:
        occupancy=1.0
    output=[]
    for nocc in noccs:
        cur_mo_occ=np.zeros((naos,))
        cur_mo_occ[:nocc]=occupancy
        output.append(cur_mo_occ)
    return np.array(output)

def read_xyz_coords(xyz_name):
    num_atoms=None
    with open(xyz_name) as fp:
        for line_id, line in enumerate(fp):
            if line_id==0:
                num_atoms=int(line)
                coords=np.zeros((num_atoms, 3))
            else:
                if line_id>1:
                    atom_id=line_id-2
                    coords[atom_id, :]=np.array([np.float(str_num) for str_num in line.split()[1:4]])
                    if atom_id==num_atoms-1:
                        return coords

def process_molpro_output(input_filename, opt_geom, restricted_method):
    basis_data_start=["BASIS", "DATA"]
    basis_data_end=["NUCLEAR", "CHARGE:"]

    ao_labels=[]
    ao_atom_ids=[]

    reading_basis=False

    final_calc=(not opt_geom)

    with open(input_filename) as fp:
        for line in fp:
            l=line.split()
            if check_els(l, basis_data_start):
                reading_basis=True
                continue
            if reading_basis:
                if check_els(l, basis_data_end):
                    cur_lower_bound=0
                    cur_atom=ao_atom_ids[0]
                    ao_atom_ranges=[]

                    for ao_id, atom_id in enumerate(ao_atom_ids):
                        if atom_id != cur_atom:
                            ao_atom_ranges.append([cur_lower_bound, ao_id])
                            cur_lower_bound=ao_id
                            cur_atom=atom_id
                    ao_atom_ranges.append([cur_lower_bound, len(ao_atom_ids)])
                    reading_basis=False
                if len(l)==6:
                    ao_labels.append(l[3])
                    ao_atom_ids.append(l[2])
            else:
                if not final_calc:
                    if check_els(l, ["END", "OF", "GEOMETRY", "OPTIMIZATION."]):
                        final_calc=True
                    else:
                        continue
                if restricted_method:
                    if check_els(l, ["Final", "occupancy:"]):
                        nocc=[int(l[2])]
                else:
                    if check_els(l, ["Final", "alpha", "occupancy:"]):
                        nocc=[int(l[3])]
                    else:
                        if check_els(l, ["Final", "beta", "occupancy:"]):
                            nocc.append(int(l[3]))
                if len(l)==5:
                    if l[3]=="Energy":
                        tot_energy=np.float(l[4])
                        return ao_labels, ao_atom_ranges, tot_energy, nocc

def get_molpro_output_matrices(filename, dim, aliases):
    matrix_start=["#", "MATRIX"]
    matrices={"ibo_mat" : None, "iao_mat" : None}

    reading_matrix=False
    output_matrix=np.zeros((dim, dim))
    cur_row=0
    cur_column=0
    with open(filename) as fp:
        for line in fp:
            l=line.split()
            if check_els(l, matrix_start):
                reading_matrix=True
                output_matrix=np.zeros((dim, dim))
                cur_row=0
                cur_column=0
                cur_name=l[2]
                continue
            if reading_matrix:
                for str_number in l:
                    output_matrix[cur_row, cur_column]=np.float(str_number.split(',')[0])
                    cur_column+=1
                    if (cur_column==dim):
                        cur_row+=1
                        cur_column=0
                if (cur_column==0) and (cur_row==dim):
                    reading_matrix=False
                    if aliases[cur_name] in matrices:
                        matrices[aliases[cur_name]].append(output_matrix)
                    else:
                        matrices[aliases[cur_name]]=[output_matrix]
    return matrices

def get_orb_ens(F, orbs):
    all_orb_ens=[]
    for spin_F, spin_orbs in zip(F, orbs):
        orb_ens=np.empty((len(spin_orbs),))
        for orb_id, orb_coeffs in enumerate(spin_orbs.T):
            orb_ens[orb_id]=np.dot(orb_coeffs,np.matmul(spin_F, orb_coeffs))
        all_orb_ens.append(orb_ens)
    return np.array(all_orb_ens)

def add_molpro_write_line(mat_name, filename, status='append'):
    return "write,"+mat_name+","+filename+","+status+",scientific\n"

def molpro_mol_header(oml_compound, method_string, opt_savexyz=None):
    header_contents='''***,molpro_calc
print,basis
symmetry,nosym;orient,noorient
'''
    for (set_var, set_val) in [("charge", oml_compound.charge), ("spin", oml_compound.charge%2)]:
        header_contents+="set,"+set_var+"="+str(set_val)+'\n'
    header_contents+='geometry={\n'
    for atom_type, coords in zip(oml_compound.atomtypes, oml_compound.coordinates):
        header_contents+=atom_type
        for coord in coords:
            header_contents+=' '+str(coord)
        header_contents+='\n'
    header_contents+='}\nbasis='+oml_compound.basis+'\n'
    if oml_compound.optimize_geometry:
        header_contents+=method_string+"\noptg,savexyz="+opt_savexyz+"\n"
    header_contents+="{"+method_string+";orbital,5100.2}\n"
    return header_contents

def get_molpro_out_processing(out_name, matrices_name, opt_savexyz, matrix_aliases, optimize_geometry, restricted_method):
    ao_labels, atom_ao_ranges, tot_energy, nocc=process_molpro_output(out_name, optimize_geometry, restricted_method)
    naos=len(ao_labels)
    results=get_molpro_output_matrices(matrices_name, naos, matrix_aliases)
    results["e_tot"]=tot_energy
    results["atom_ao_ranges"]=atom_ao_ranges
    results["aos"]=[]
    for atom_id, atom_ao_range in enumerate(atom_ao_ranges):
        for ao_id in range(*atom_ao_range):
            results["aos"].append(AO(ao_labels[ao_id], atom_id=atom_id))
    results["mo_energy"]=get_orb_ens(results["fock_mat"], results["mo_coeff"])
    results["iao_mat"]=None
    results["mo_occ"]=mo_occs(nocc, naos)
    if optimize_geometry:
        results["opt_coords"]=read_xyz_coords(opt_savexyz)
    return results

def make_tmpfiles():
    tmpdir=mktmpdir()
    inp_name="molpro.inp"
    out_name="molpro.out"
    matrices_name="matrices.dat"
    opt_savexyz="test_opt.xyz"
    return inp_name, out_name, matrices_name, opt_savexyz, tmpdir

def get_molpro_calc_data_HF(oml_compound):
    inp_name, out_name, matrices_name, opt_savexyz, tmpdir=make_tmpfiles()
    os.chdir(tmpdir)
    inp_contents=molpro_mol_header(oml_compound, "hf", opt_savexyz=opt_savexyz)+'''
{matrop;
load,dscf,den,5100.2
load,s
load,f,fock,5100.2
load,orbs,orbital,5100.2
coul,j,dscf
exch,k,dscf
'''
    inp_contents+=add_molpro_write_line("orbs", matrices_name, status="new")
    for molpro_mat_name in ["j", "k", "s", "f"]:
        inp_contents+=add_molpro_write_line(molpro_mat_name, matrices_name)
    inp_contents+='''
}
---
'''
    inp_file=open(inp_name, 'w')
    inp_file.write(inp_contents)
    inp_file.close()
    subprocess.run(["molpro", inp_name, "-o", out_name])
    matrix_aliases={"F" : "fock_mat", "K" : "k_mat", "J" : "j_mat", "ORBS" : "mo_coeff", "S" : "ovlp_mat"}
    results=get_molpro_out_processing(out_name, matrices_name, opt_savexyz, matrix_aliases, oml_compound.optimize_geometry, True)
    for mult_mat_keys in ["k_mat", "j_mat"]:
        results[mult_mat_keys][0]*=2.0
    os.chdir("..")
    rmdir(tmpdir)
    return results

def get_molpro_calc_data_UHF(oml_compound):
    inp_name, out_name, matrices_name, opt_savexyz, tmpdir=make_tmpfiles()
    os.chdir(tmpdir)
    inp_contents=molpro_mol_header(oml_compound, "uhf", opt_savexyz=opt_savexyz)+'''
{matrop;
load,dscf1,density,5100.2,type=charge
load,dscf2,density,5100.2,type=spin
load,h0,h0
add,dalpha,1,dscf1,1,dscf2
add,dbeta,1,dscf1,-1,dscf2
load,s
load,orbs1,orbital,5100.2,set=1
load,orbs2,orbital,5100.2,set=2
coul,j1,dalpha
coul,j2,dbeta
exch,k1,dalpha
exch,k2,dbeta
add,f1,1,h0,1,j1,1,j2,-1,k1
add,f2,1,h0,1,j1,1,j2,-1,k2
'''
    inp_contents+=add_molpro_write_line("orbs1", matrices_name, status="new")
    for molpro_mat_name in ["orbs2", "j1", "j2", "k1", "k2", "f1", "f2", "s"]:
        inp_contents+=add_molpro_write_line(molpro_mat_name, matrices_name)
    inp_contents+='''
}
---
'''
    inp_file=open(inp_name, 'w')
    inp_file.write(inp_contents)
    inp_file.close()
    subprocess.run(["molpro", inp_name, "-o", out_name])
    matrix_aliases={"F1" : "fock_mat", "F2" : "fock_mat", "K1" : "k_mat", "K2" : "k_mat",
                    "J1" : "j_mat", "J2" : "j_mat", "ORBS1" : "mo_coeff", "ORBS2" : "mo_coeff",
                    "S" : "ovlp_mat"}
    results=get_molpro_out_processing(out_name, matrices_name, opt_savexyz, matrix_aliases, oml_compound.optimize_geometry, False)
    os.chdir("..")
    rmdir(tmpdir)
    return results


get_molpro_calc_data={"HF" : get_molpro_calc_data_HF, "UHF" : get_molpro_calc_data_UHF}
