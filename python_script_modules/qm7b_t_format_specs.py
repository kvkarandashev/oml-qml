from qml.oml_compound import OML_compound
import subprocess, glob

class QuantityNotAvailableError(Exception):
    pass


from urllib.parse import quote_plus

from qml.oml_compound import OML_compound

def_float_format='{:.8E}'

au_to_kcalmol_mult=627.50960803

def potential_energy(xyz_name, **oml_calc_kwargs):
    oml_comp=OML_compound(xyz = xyz_name, mats_savefile = xyz_name, **oml_calc_kwargs)
    oml_comp.run_calcs()
    return oml_comp.e_tot*au_to_kcalmol_mult

class Quantity:
    def __init__(self, quant_name):
        self.name=quant_name
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
        if output is None:
            raise QuantityNotAvailableError
        else:
            return output
    def OML_calc_quant(self, xyz_name, **oml_calc_kwargs):
        return potential_energy(xyz_name, **oml_calc_kwargs)

class Quantity_diff:
    def __init__(self, quant_name1, quant_name2):
        self.quant1=Quantity(quant_name1)
        self.quant2=Quantity(quant_name2)
    def extract_xyz(self, filename):
        return self.quant2.extract_xyz(filename)-self.quant1.extract_xyz(filename)
    # Perhaps should be improved.
    def OML_calc_quant(self, xyz_name, **oml_calc_kwargs):
        return self.quant2.OML_calc_quant(xyz_name, **oml_calc_kwargs)-self.quant1.OML_calc_quant(xyz_name, **oml_calc_kwargs)
        

def file_to_2Dlist(filename):
    inp=open(filename, 'r')
    return [l.split(',') for l in inp.read().split('\n')]

def create_xyz_files(QM7bT_dir, output_dir):
    val_lines = file_to_2Dlist(QM7bT_dir+"/values.csv")
    value_names=val_lines[0]
    for val_row in val_lines[1:]:
        xyz_id=val_row[0]
        new_xyz_name=output_dir+'/ext_'+xyz_id+'.xyz'
        possible_xyzs=glob.glob(QM7bT_dir+"/molecules/"+xyz_id+'__*.xyz')
        if len(possible_xyzs)>1:
            print("xyz id duplicated:", xyz_id, possible_xyzs)
            quit()
        elif len(possible_xyzs)==0:
            print('xyz id absent', xyz_id)
            quit()
        subprocess.run(["cp", '-f', possible_xyzs[0], new_xyz_name])
        cur_xyz=open(new_xyz_name, "a+")
        for val, val_name in zip(val_row, value_names):
            if val_name != '':
                cur_xyz.write(val_name+" : "+str(val)+'\n')
        cur_xyz.close()
    val_csvfile.close()
    entry_csvfile.close()
            
