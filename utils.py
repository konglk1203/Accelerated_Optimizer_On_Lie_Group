import torch
import math
from scipy.linalg import logm

def parameter_HB(mu, L):
    h=math.sqrt(mu)/(4*L)
    gamma=2*math.sqrt(mu)
    c=mu/(16*L)
    return {'h':h, 'gamma':gamma, 'c':c}


def parameter_NAG_SC(mu, L):
    h=1/math.sqrt(2*L)
    gamma=2*math.sqrt(mu)/(1-torch.sqrt(mu)*h)
    c=math.sqrt(mu/(2*L))/30
    return {'h':h, 'gamma':gamma, 'c':c}


def parameter_momentumless(mu, L):
    h=1/L
    c=mu/L
    return {'h':h, 'c':c}

@torch.no_grad()
def lyap_HB(parameter_dict):
    U=parameter_dict['U']
    g=parameter_dict['g']
    xi=parameter_dict['xi']
    g_star=parameter_dict['g_star']
    g_last=parameter_dict['g_last']
    h=parameter_dict['h']
    gamma=parameter_dict['gamma']
    # return 1/(1-gamma*h)*(U(g_last)-U(g_star))+torch.sum(xi**2)/4+torch.sum((gamma/(1-gamma*h)*scipy.linalg.logm(torch.linalg.inv(g_star)@g).real+xi)**2)/4
    return 1/(1-gamma*h)*(U(g_last)-U(g_star))+torch.sum(xi**2)/8+torch.sum((gamma/(1-gamma*h)*logm(torch.linalg.inv(g_star)@g).real+xi)**2)/8
    # return U(g)-U(g_star)+(1-gamma*h)**2*torch.sum(xi**2)/4



@torch.no_grad()
def lyap_NAG_SC(parameter_dict):
    U=parameter_dict['U']
    g=parameter_dict['g']
    xi=parameter_dict['xi']
    g_star=parameter_dict['g_star']
    g_last=parameter_dict['g_last']
    nabla_g_last=parameter_dict['nabla_g_last']
    h=parameter_dict['h']
    gamma=parameter_dict['gamma']
    return 1/(1-gamma*h)*(U(g_last)-U(g_star))+torch.sum(xi**2)/8+torch.sum((gamma/(1-gamma*h)*logm(torch.linalg.inv(g_star)@g).real+xi+h*nabla_g_last)**2)/8-h**2*(2-gamma*h)/4/(1-gamma*h)*torch.sum(nabla_g_last**2)/2


