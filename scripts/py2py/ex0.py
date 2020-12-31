from pyccel.decorators import types

@types( 'int' )
def f( n ):

    x = [i for i in range(n)]
