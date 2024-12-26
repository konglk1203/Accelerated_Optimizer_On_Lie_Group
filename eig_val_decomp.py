import torch
import numpy as np


def generate_eig_value_random_normal(n):
    A=torch.randn(n, n)
    A=(A+A.t())/2/np.sqrt(n)
    eig_vals, eig_vecs=torch.linalg.eigh(A)
    return eig_vals

def generate_eig_value_artificial_conditional_number(n, kappa):
    kappa=float(kappa)
    assert kappa>(n-1)**2
    eig_vals=torch.arange(n, dtype=torch.get_default_dtype())
    eig_vals[-1]=kappa/(n-1)
    return eig_vals


def eig_val_decomp_problem(eig_vals, device='cpu', dtype=None):
    if dtype==None:
        dtype=torch.get_default_dtype()
    n=len(eig_vals)
    eig_vals=eig_vals.to(device=device, dtype=dtype)
    eig_vecs=torch.zeros(n, n, device=device, dtype=dtype)
    torch.nn.init.orthogonal_(eig_vecs)
    A=eig_vecs@torch.diag(eig_vals)@eig_vecs.T
    min_val=torch.sum(eig_vals*torch.arange(n))
    L=(eig_vals[-1]-eig_vals[0])*(n-1)
    mu=torch.min(eig_vals[1:]-eig_vals[:-1])
    sol_dict={'min_val':-min_val, 'L':L, 'mu':mu, 'X_sol': eig_vecs}
    return A, sol_dict


def eig_val_decomp_loss(A, X):
    n=A.shape[0]
    D=torch.diag(torch.arange(n)).to(device=A.device, dtype=A.dtype)
    return -torch.trace(X.T@A@X@D)
