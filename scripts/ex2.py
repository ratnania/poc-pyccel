from pyccel.decorators import types

@types('int', 'int[:]')
def f(n, xs):
    for i in range(4):
        xs[i] = 2
