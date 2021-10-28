# This short script demonstrates that analytical derivatives implemented for the
# gaussian_pos_sum_restr_kernel function match the ones obtained with finite difference.

from qml.kernels_wders import gaussian_pos_sum_restr_kernel_wders
import random
import numpy as np

random.seed(3)

dimf=3

def random_vec():
    output=[]
    for q in range(dimf):
        output.append(random.random())
    return np.array(output)

def random_vecs(nvecs):
    return [random_vec() for l in range(nvecs)]

sigma_prop_coeff=0.5

sigmas=np.array([float(i+1)*sigma_prop_coeff for i in range(dimf)])

ninit=4
nother=3
#ninit=1
#nother=0

A=random_vecs(ninit)
B=random_vecs(nother)

B.append(A[0]*1.5)
B.append(B[0]*2.5)

A=np.array(A)
B=np.array(B)

print("A:", A)
print("B:", B)

mat0=gaussian_pos_sum_restr_kernel_wders(A, B, sigmas)

print("mat:", mat0)

fd_step=0.00001

mat_analytic_der=gaussian_pos_sum_restr_kernel_wders(A, B, sigmas, with_ders=True)

print(mat_analytic_der)

for der_id in range(dimf):
    mat_fd=np.zeros(mat0.shape)
    for fd in [-1, 1]:
        cur_sigmas=np.copy(sigmas)
        cur_sigmas[der_id]+=fd_step*fd
        mat_fd+=fd*gaussian_pos_sum_restr_kernel_wders(A, B, cur_sigmas)
    mat_fd/=2*fd_step
#    print("mat_fd", mat_fd)
    print("analytic", mat_analytic_der[:, :, der_id+1])
    print("diff", mat_fd-mat_analytic_der[:, :, der_id+1])
