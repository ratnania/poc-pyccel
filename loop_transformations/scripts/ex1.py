from pyccel.decorators import types

@types('int', 'int[:]')
def f(n, xs):
    for i in range(n):
        xs[i] = i
