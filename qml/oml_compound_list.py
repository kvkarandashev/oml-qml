# MIT License
#
# Copyright (c) 2016-2017 Anders Steen Christensen, Felix Faber, Lars Andersen Bratholm
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from .oml_compound import OML_compound, OML_Slater_pair
from .python_parallelization import embarassingly_parallel
import os

class OML_compound_list(list):
    """ The class was created to allow easy embarassing parallelization of operations with lists of OML_compound objects.
    """
    def run_calcs(self, disable_openmp=True):
        self.embarassingly_parallelize(after_run_calcs, disable_openmp=disable_openmp)
    def generate_orb_reps(self, rep_params, disable_openmp=True):
        self.embarassingly_parallelize(after_gen_orb_reps, rep_params, disable_openmp=disable_openmp)
    def embarassingly_parallelize(self, func_in, other_args, disable_openmp=True):
        new_vals=embarassingly_parallel(func_in, self, other_args, disable_openmp=disable_openmp)
        for i in range(len(self)):
            self[i]=new_vals[i]

#   Both functions are dirty as they modify the arguments, but it doesn't matter in this particular case.
def after_run_calcs(oml_comp):
    oml_comp.run_calcs()
    return oml_comp

def after_gen_orb_reps(oml_comp, rep_params):
    oml_comp.generate_orb_reps(rep_params)
    return oml_comp

def OML_compound_list_from_xyzs(xyz_files, **oml_comp_kwargs):
    return OML_compound_list([OML_compound(xyz = xyz_file, mats_savefile = xyz_file, **oml_comp_kwargs) for xyz_file in xyz_files])
    
def OML_Slater_pair_list_from_xyzs(xyz_files, **slater_pair_kwargs):
    return OML_compound_list([OML_Slater_pair(xyz = xyz_file, mats_savefile = xyz_file, **slater_pair_kwargs) for xyz_file in xyz_files])
