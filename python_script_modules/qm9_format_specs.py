def HOMO_en(oml_comp):
    return oml_comp.HOMO_en()
    
def LUMO_en(oml_comp):
    return oml_comp.LUMO_en()

quant_properties = {'Dipole moment' : (6, 'Debye'),
                'Isotropic polarizability' : (7, 'Bohr^3'),
                'HOMO eigenvalue': (8, 'Hartree', HOMO_en),
                'LUMO eigenvalue': (9, 'Hartree', LUMO_en),
                'HOMO-LUMO gap': (10, 'Hartree'),
                'Electronic spacial extent': (11, 'Bohr^2'),
                'Zero point vibrational energy': (12, 'Hartree'),
                'Internal energy at 0 K': (13, 'Hartree'),
                'Internal energy at 298.15 K': (14, 'Hartree'),
                'Enthalpy at 298.15 K': (15, 'Hartree'),
                'Free energy at 298.15 K': (16, 'Hartree'),
                'Heat capacity at 298.15 K': (17, 'cal/(mol K)'),
                'Highest vibrational frequency': (18, 'cm^-1')}

class Quantity:
    def __init__(self, quant_name):
        self.name=quant_name
        self.qm9_id=quant_properties[quant_name][0]
        self.dimensionality=quant_properties[quant_name][1]
    def extract_xyz(self, filename):
        file=open(filename, 'r')
        lines=file.readlines()
        output=None
        if self.name == 'Highest vibrational frequency':
            first_line_passing=True
        for l in lines:
            lsplit=l.split()
            if self.name == 'Highest vibrational frequency':
                if first_line_passing:
                    first_line_passing=False
                    continue
                try:
                    output=max([float(freq_str) for freq_str in lsplit]) # this will fail for all lines but for the one with molecule number (first line) and frequencies
                    break
                except:
                    continue
            else:
                if lsplit[0] == "gdb":
                    output=float(lsplit[self.qm9_id-1])
                    break
        file.close()
        return output
    def extract_byprod_result(self, filename):
        file=open(filename, 'r')
        lines=file.readlines()
        output=None
        for l in lines:
            lsplit=l.split()
            if int(lsplit[0]) == self.qm9_id:
                output=float(lsplit[1])
                break
        file.close()
        return output
    def OML_comp_extract_byprod(self, oml_compound):
        return quant_properties[self.name][2](oml_compound)
    def write_byprod_result(self, val, io_out):
        io_out.write(str(self.qm9_id)+" "+str(val)+"\n")
