import torch

def sobolev(x, t, alpha=1.0, beta=1.0, keepbatch=False, pointwise=False):
     # t is the time
     # assumes that x is already padded.
     # x is of shape (batch_size, channels, n1, n2 (, n3))
     assert x.dim() in [4,5]

     grads = torch.gradient( x, dim=list(range(2, x.dim())) )
     grad_sqr_norm = sum( grad.pow(2) for grad in grads )
     norm = alpha * x.pow(2) + beta * t.view(-1,*[1]*(x.dim()-1)) * grad_sqr_norm
     # norm = alpha * x.pow(2) + beta * grad_sqr_norm
     if pointwise: return norm
     if keepbatch: return norm.mean(dim=list(range(1, x.dim())))
     return norm.mean()

