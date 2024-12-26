# --------------------------------------------------------
# This is an implementation for Variational Stiefel SGD/Adam in the paper
# Momentum Stiefel Optimizer, with Applications to
# Suitably-Orthogonal Attention, and Optimal Transport (ICLR 2023)
# https://arxiv.org/pdf/2205.14173.pdf
# --------------------------------------------------------

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
import torch.nn as nn
import numpy as np
import math
from torch import Tensor
from typing import List, Optional
from utils_SOn_Optimizers import *


inner_prod_param_dict={'Canonical':0.5, 'Euclidean':0.0}



def _update_func(X, Y, V, X_grad, grad_Y_last, grad_V_last, NAG_SC, lr, momentum, dampening):
    Xt_Xgrad=torch.matmul(X.t(), X_grad)
    grad_Y=Xt_Xgrad-Xt_Xgrad.t()
    if NAG_SC:
        if grad_Y_last.nelement() == 0:
            grad_Y_last.copy_(grad_Y)
        grad_Y_NAG_SC=momentum*(grad_Y-grad_Y_last)+grad_Y
        # X_grad_NAG_SC=X_grad
        grad_Y_last.copy_(grad_Y)
        grad_Y=grad_Y_NAG_SC
    # Dynamics phi_1
    Y.mul_(momentum).add_(grad_Y,alpha=-(1-dampening))
    X.copy_(X.matmul(torch.matrix_exp(lr*Y)))

class MomentumOptimizer_LieGroup_SOn(Optimizer):
    r""" Implementation of Momentum Stiefel (S)GD from the paper
    Momentum Stiefel Optimizer, with Applications to Suitably-Orthogonal Attention, and Optimal Transport (https://arxiv.org/abs/2205.14173)

    Purpose:
        Given a function f(X), find the minimum value of f under constraint that X has orthonormal columns
    Args:
        - params: A list of matrices. Containing parameters to optimize. 
        - lr: learning rate
        - momentum (float, optional): momentum factor (default: 0.9)
        - dampening (float, optional): dampening for momentum (default: 0)
    Discussion: 
        - We recommend using the same hyperparameters when the model contains both Euclidean parameters and Stiefel parameters. See Remark 1 in the paper for details.
        - The matrices being optimized should have number of rows >= number of columns. Otherwise, the matrix will be transposed without warning. For tensors with more than 2 dimensions, all the dimensions will be flattened excepted the first dimension to create a matrix.
        - No special orthonormal initialization for Stiefel matrices is required. Commonly used element-wise random Gaussian matrices will work and our optimizer will automatically project it onto the Stiefel manifold. However, explicit initialization using `torch.nn.init.orthogonal_` is still recommended.
    """
    def __init__(self, params, lr=required, momentum=0.9, dampening=0, NAG_SC=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        # metric parameter in Definition 1 in the paper

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, NAG_SC=NAG_SC)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            dampening = group['dampening']
            NAG_SC=group['NAG_SC']
            for X_raw in group['params']:
                if X_raw.grad is None:
                    continue
                # If X has more than 2 dimensions, all the dimensions except the first one will be flattened to make it a matrix.
                X=X_raw.view(-1,X_raw.shape[-2], X_raw.shape[-1])
                X_grad=X_raw.grad.view(-1,X_raw.shape[-2], X_raw.shape[-1])
                # X should be a tall and thin matrix (n>m). Otherwise, it will be transposed.

                assert X.shape[-2]==X.shape[-1]

                # Make the algorithm compatible with SO(n)
                # In that case, n=m, and we no longer need V                    
                param_state = self.state[X_raw]
                # Same notation as in the paper are used for Y,V.
                if 'Y_buffer' not in param_state:
                    Y = param_state['Y_buffer']=torch.zeros(X.shape[0], X.shape[-1],X.shape[-1], device=X.device, dtype=X.dtype)
                Y = param_state['Y_buffer']
                
                if 'grad_Y_last' not in param_state:
                    if NAG_SC:
                        param_state['grad_Y_last']=torch.empty_like(param_state['Y_buffer'])
                    else:
                        param_state['grad_Y_last']=None
                grad_Y_last=param_state['grad_Y_last']
                if NAG_SC:
                    update_func=lambda X, Y, X_grad, grad_Y_last: _update_func(X, Y, None, X_grad, grad_Y_last, None, NAG_SC, lr, momentum, dampening)
                    torch.vmap(update_func, out_dims=None)(X, Y, X_grad, grad_Y_last)
                else:
                    update_func=lambda X, Y, X_grad: _update_func(X, Y, None, X_grad, None, None, NAG_SC, lr, momentum, dampening)
                    torch.vmap(update_func, out_dims=None)(X, Y, X_grad)
                
                # Check the structure for tangent bundle. For debug only. Please comment out.
                # assert torch.norm(X.t()@X-torch.eye(m, dtype=X.dtype, device=X.device))<torch.finfo(X.dtype).eps*torch.numel(Y)*10
                # assert torch.norm(Y.t()+Y)<torch.finfo(X.dtype).eps*torch.numel(Y)*10
        return loss




class CombinedOptimizer(torch.optim.Optimizer):
    r"""
        This can be used when Euclidean and Stiefel parameters are contained in one model and are being optimized at the same time.
        This is due to that our StiefelSGD and Euclidean SGD (StiefelAdam and Euclidean Adam) uses the same hyperparameters and do not need to be tuned separately.
    """
    def __init__(self, *arg):
        self.optimizer_list=list(arg)
        param_group=[]
        for op in self.optimizer_list:
            if op==None:
                continue
            for pg in op.param_groups:
                param_group.append(pg)
        super().__init__(param_group, defaults=dict())
    def zero_grad(self, set_to_none: bool = False):
        for op in self.optimizer_list:
            if op==None:
                continue
            op.zero_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for op in self.optimizer_list:
            if op==None:
                continue
            loss=op.step()
        return loss
