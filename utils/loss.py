import numpy as np
import torch
import torch.nn as nn

def Dice(output, target, eps=1e-5):
    target = target.float()
    num = 2 * (output * target).sum()
    den = output.sum() + target.sum() + eps
    return num/den

def KL_Divergence(mu1, sigma1, mu2, sigma2):
    K = mu1.size(-1)
    sigma2_inv = torch.inverse(sigma2)
    trace_term = torch.trace(torch.matmul(sigma2_inv,sigma1))

    diff = mu2 - mu1
    mid_term = torch.matmul(torch.matmul(diff.unsqueeze(-1).transpose(-1,-2), sigma2_inv), diff.unsqueeze(-1)).squeeze()

    log_term = torch.log(torch.det(sigma2) / torch.det(sigma1))
    KL_D = 0.5 * (trace_term + mid_term + log_term - K)
    return KL_D
