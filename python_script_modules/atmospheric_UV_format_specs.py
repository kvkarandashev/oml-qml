from qml.oml_compound import OML_compound

def HOMO_en(xyz_name, calc_type="HF", basis="sto-3g", dft_xc='lda,vwn', dft_nlc=''):
    oml_comp=OML_compound(xyz = xyz_name, mats_savefile = xyz_name, calc_type=calc_type, basis=basis, dft_xc=dft_xc, dft_nlc=dft_nlc)
    oml_comp.run_calcs()
    return oml_comp.HOMO_en()
    
def LUMO_en(xyz_name, calc_type="HF", basis="sto-3g", dft_xc='lda,vwn', dft_nlc=''):
    oml_comp=OML_compound(xyz = xyz_name, mats_savefile = xyz_name, calc_type=calc_type, basis=basis, dft_xc=dft_xc, dft_nlc=dft_nlc)
    oml_comp.run_calcs()
    return oml_comp.LUMO_en()

def first_excitation(xyz_name, calc_type="HF", basis="sto-3g", dft_xc='lda,vwn', dft_nlc=''):
    return LUMO_en(xyz_name, calc_type=calc_type, basis=basis, dft_xc=dft_xc, dft_nlc=dft_nlc)-HOMO_en(xyz_name, calc_type=calc_type, basis=basis, dft_xc=dft_xc, dft_nlc=dft_nlc)

def potential_energy(xyz_name, calc_type="HF", basis="sto-3g", dft_xc='lda,vwn', dft_nlc=''):
    oml_comp=OML_compound(xyz = xyz_name, mats_savefile = xyz_name, calc_type=calc_type, basis=basis, dft_xc=dft_xc, dft_nlc=dft_nlc)
    oml_comp.run_calcs()
    return oml_comp.e_tot

def blank_function(*args, **kwargs):
    return 0.0


quant_properties = {'ground_state_energy' : (0, 'Hartree', potential_energy),
                'S1_excitation' : (1, 'Hartree', first_excitation),
                'S1_oscillator_strength' : (2, 'Hartree', blank_function),
                'S2_excitation' : (3, 'Hartree', blank_function),
                'S2_oscillator_strength' : (4, 'Hartree', blank_function),
                'S3_excitation' : (5, 'Hartree', blank_function),
                'S3_oscillator_strength' : (6, 'Hartree', blank_function)}


class Quantity:
    def __init__(self, quant_name):
        self.name=quant_name
        self.quant_pos=quant_properties[quant_name][0]
        self.dimensionality=quant_properties[quant_name][1]
    def extract_xyz(self, filename):
        file=open(filename, 'r')
        l=file.readlines()[1]
        file.close()
        return float(l.split()[self.quant_pos])
    def OML_calc_quant(self, xyz_name, calc_type="HF", basis="sto-3g", dft_xc='lda,vwn', dft_nlc=''):
        return quant_properties[self.name][2](xyz_name, calc_type=calc_type, basis=basis, dft_xc=dft_xc, dft_nlc=dft_nlc)
    def write_byprod_result(self, val, io_out):
        io_out.write(str(self.qm9_id)+" "+str(val)+"\n")
