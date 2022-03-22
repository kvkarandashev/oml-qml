# Miscellaneous functions and classes.
import pickle, subprocess

def dump2pkl(obj, filename):
    output_file = open(filename, "wb")
    pickle.dump(obj, output_file)
    output_file.close()

def loadpkl(filename):
    input_file = open(filename, "rb")
    obj=pickle.load(input_file)
    input_file.close()
    return obj
    
def mktmpdir():
    return subprocess.check_output(["mktemp", "-d", "-p", "."], text=True).rstrip("\n")
    
def rmdir(dirname):
    subprocess.run(["rm", "-Rf", dirname])

def write_compound_to_xyz_file(compound, xyz_file_name):
    write_xyz_file(compound.coordinates, compound.atomtypes, xyz_file_name)

def write_xyz_file(coordinates, elements, xyz_file_name):
    xyz_file=open(xyz_file_name, 'w')
    xyz_file.write(str(len(coordinates))+'\n\n')
    for atom_coords, element in zip(coordinates, elements):
        xyz_file.write(element+" "+' '.join([str(atom_coord) for atom_coord in atom_coords])+'\n')
    xyz_file.close()

def write_bytes(byte_string, file_name):
    file_output=open(file_name, 'wb')
    print(byte_string, file=file_output)
    file_output.close()

class OptionUnavailableError(Exception):
    pass

