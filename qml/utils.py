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
