from pyccel.decorators import types

@types('int[:,:,:]')
def f(xs):
    from numpy import shape
    n,m,p = shape(xs)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                xs[i,j,k] = 2
