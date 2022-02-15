# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

# Note that we need to change the directory for tests involving the import
# statement

import os

from pyccel.codegen.printing.pycode import pycode

from loops import Transform

# **********************************************************************************
def test_split_rank_1(fname, **kwargs):
    inner_unroll = kwargs.pop('inner_unroll', False)

    T = Transform(fname)
    f = T.split({'index': 'i', 'size': 4, 'inner_unroll': inner_unroll})

    print('****************** BEFORE ******************')
    code = pycode(T.func)
    print(code)

    print('****************** AFTER  ******************')
    code = pycode(f)
    print(code)

# **********************************************************************************
def test_split_rank_2(fname, **kwargs):
    inner_unroll = kwargs.pop('inner_unroll', False)

    T = Transform(fname)

    f = T.split({'index': 'i', 'size': 2, 'inner_unroll': inner_unroll},
                {'index': 'j', 'size': 4, 'inner_unroll': inner_unroll})

    print('****************** BEFORE ******************')
    code = pycode(T.func)
    print(code)

    print('****************** AFTER  ******************')
    code = pycode(f)
    print(code)

# **********************************************************************************
def test_split_rank_3(fname, **kwargs):
    inner_unroll = kwargs.pop('inner_unroll', False)

    T = Transform(fname)

    f = T.split({'index': 'i', 'size': 2, 'inner_unroll': inner_unroll},
                {'index': 'j', 'size': 3, 'inner_unroll': inner_unroll},
                {'index': 'k', 'size': 4, 'inner_unroll': inner_unroll})

    print('****************** BEFORE ******************')
    code = pycode(T.func)
    print(code)

    print('****************** AFTER  ******************')
    code = pycode(f)
    print(code)

# **********************************************************************************
from pyccel.parser.utilities import read_file

def run_tests():
    # ...
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, 'scripts')

    files = sorted(os.listdir(path_dir))
    files = [os.path.join(path_dir,f) for f in files if (f.endswith(".py"))]
    # ...

    for fname in files:
        print('>>>>>> testing {0}'.format(str(os.path.basename(fname))))
        code = read_file(fname)
        print('****************** BEFORE ******************')
        print(code)

######################
if __name__ == '__main__':
#    run_tests()

#    test_split_rank_1('scripts/ex1.py', inner_unroll=False)
    test_split_rank_1('scripts/ex1.py', inner_unroll=True)

#    test_split_rank_2('scripts/ex2.py', inner_unroll=False)
#    test_split_rank_2('scripts/ex2.py', inner_unroll=True)

#    test_split_rank_3('scripts/ex3.py', inner_unroll=False)
#    test_split_rank_3('scripts/ex3.py', inner_unroll=True)
