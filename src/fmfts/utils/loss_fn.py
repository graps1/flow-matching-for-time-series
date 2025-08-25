import torch

def sobolev(x, t, alpha=1.0, beta=20.0, keepbatch=False, pointwise=False):
     # t is the time
     # assumes that x is already padded.
     # x is of shape (batch_size, channels, n1, n2 (, n3))
     assert x.dim() in [4,5]


     grads = torch.gradient( x, dim=list(range(2, x.dim())) )
     grad_sqr_norm = sum( grad.pow(2) for grad in grads )
     norm = alpha * x.pow(2) + beta * (t**2).view(-1,*[1]*(x.dim()-1)) * grad_sqr_norm
     if pointwise: return norm
     if keepbatch: return norm.mean(dim=list(range(1, x.dim())))
     return norm.mean()


     # if len(x.shape) == 4:
     #      grad = torch.gradient(x, dim=(-))
     #      # dx = x[:,:,1:,:] - x[:,:,:-1,1:]
     #      # dy = x[:,:,:,1:] - x[:,:,:,:-1]
     #      # grad = torch.gradient([dx, dy], dim=-1)
     #      # grad_sqr_norm = dx.pow(2).mean(dim=[1,2,3]) + dy.pow(2).mean(dim=[1,2,3])
     #      # grad_sqr_norm = t**2 * grad_sqr_norm
     #      result = alpha * x.pow(2).mean(dim=[1,2,3]) + beta * grad_sqr_norm
     #      if not keepbatch: result = result.mean()
     #      return result

     # if len(x.shape) == 5:
     #      dx = x[:,:,1:,:,:] - x[:,:,:-1,:,:] 
     #      dy = x[:,:,:,1:,:] - x[:,:,:,:-1,:]
     #      dz = x[:,:,:,:,1:] - x[:,:,:,:,:-1]
     #      grad_sqr_norm = dx.pow(2).mean(dim=[1,2,3,4]) + dy.pow(2).mean(dim=[1,2,3,4]) + dz.pow(2).mean(dim=[1,2,3,4])
     #      grad_sqr_norm = t**2 * grad_sqr_norm
     #      result = alpha * x.pow(2).mean(dim=[1,2,3,4]) + beta * grad_sqr_norm
     #      if not keepbatch: result = result.mean()
     #      return result

     #  grad_sqr_norm = 0
     #  for d in range(2, x.dim()):
     #      s1 = [slice(0, x.shape[j]  , 1) for j in range(d)           ] + \
     #           [slice(1, x.shape[d]  , 1)                             ] + \
     #           [slice(0, x.shape[j]  , 1) for j in range(d+1, x.dim())]
     #      s2 = [slice(0, x.shape[j]  , 1) for j in range(d)           ] + \
     #           [slice(0, x.shape[d]-1, 1)                             ] + \
     #           [slice(0, x.shape[j]  , 1) for j in range(d+1, x.dim())]
     #      dx = t.view(-1,*[1]*(x.dim()-1)) * (x[s1] - x[s2]) 
     #      grad_sqr_norm += dx.pow(2).mean(dims=list(range(x.dim()))[keepbatch:])
     #  return alpha * x.pow(2).mean() + beta * grad_sqr_norm