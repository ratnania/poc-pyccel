import numpy as np 
from pyccel.decorators import types
from pyccel.epyccel import epyccel

@types('int', 'real[:]', 'real[:]')
def f(n,xs,ys):
    for i in range(0,n):
        ys[i] = ys[i]+xs[i]


f = epyccel(f)
n = 5
xs = np.linspace(0,1,n)
ys = np.linspace(0,1,n)    

f(n,xs,ys)

