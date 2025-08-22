
def sobolev(x, t, alpha=1.0, beta=1.0):
    # t is the time
    # assumes that x is already padded.
    # x is of shape (batch_size, channels, n1, n2, n3, ...)
    grad_sqr_norm = 0
    for d in range(2, x.dim()):
        s1 = [slice(0, x.shape[j]  , 1) for j in range(d)           ] + \
             [slice(1, x.shape[d]  , 1)                             ] + \
             [slice(0, x.shape[j]  , 1) for j in range(d+1, x.dim())]
        s2 = [slice(0, x.shape[j]  , 1) for j in range(d)           ] + \
             [slice(0, x.shape[d]-1, 1)                             ] + \
             [slice(0, x.shape[j]  , 1) for j in range(d+1, x.dim())]
        dx = t.view(-1,*[1]*(x.dim()-1)) * (x[s1] - x[s2]) * x.shape[d]
        grad_sqr_norm += dx.pow(2).mean()
    return alpha * x.pow(2).mean() + beta * grad_sqr_norm