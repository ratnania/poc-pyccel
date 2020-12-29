import numpy as np 
from pyccel.decorators import types
from pyccel.epyccel import epyccel

@types('int')
def f(n):
    return n*2

f = epyccel(f)
n = 5
f(n)
