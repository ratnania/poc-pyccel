from pyccel.decorators import types

@types('int[:,:]')
def f(xs):
    from numpy import shape
    n,m = shape(xs)
    for i in range(n):
        for j in range(m):
            xs[i,j] = i+j
