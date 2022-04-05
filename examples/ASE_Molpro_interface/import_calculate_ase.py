from glob import glob
from ase.io import extxyz
from qml.oml_compound_list import OML_compound_list_from_ASEs
import os

# Folder with xyzs.
xyz_folder="./qm9_xyz_files"

# Create a list with ASE objects.
xyz_files=glob(xyz_folder+"/*.xyz")

ase_list=[]
for xyz_file in xyz_files:
    xyz=open(xyz_file, 'r')
    ase_list+=list(extxyz.read_extxyz(xyz))
    xyz.close()


# Create an OML_compound_list object. It has the same properties as the list object,
# except it has some attributes that allow easy parallelization of calculations.
oml_compounds=OML_compound_list_from_ASEs(ase_list, software="molpro")

for i in range(len(oml_compounds)):
    oml_compounds[i].temp_calc_dir="temp_"+str(i)

# Run the calculations in parallel.
oml_compounds.run_calcs(num_threads=8, num_procs=4)


# Write names of xyz files and the corresponding energies.
print("# XYZ # total potential energy")
for xyz_file, oml_compound in zip(xyz_files, oml_compounds):
    print(xyz_file, oml_compound.e_tot)
