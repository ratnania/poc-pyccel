# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

# Note that we need to change the directory for tests involving the import
# statement

import os

from pyccel.parser.parser   import Parser
from pyccel.codegen.codegen import Codegen
from pyccel.errors.errors   import Errors
from pyccel.parser.utilities import read_file

# **********************************************************************************
def py2py(filename):
    # ... run the semantic stage
    pyccel = Parser(filename)
    ast = pyccel.parse()

    settings = {}
    ast = pyccel.annotate(**settings)

    name = os.path.basename(filename)
    name = os.path.splitext(name)[0]

    codegen = Codegen(ast, name)

    # reset Errors singleton
    errors = Errors()
    errors.reset()

    return codegen.doprint(language='python')

# **********************************************************************************
def test_py2py(fname):
    print('>>>>>> testing {0}'.format(str(os.path.basename(fname))))
    code = read_file(fname)
    print('****************** BEFORE ******************')
    print(code)
    code = py2py(fname)
    print('****************** AFTER  ******************')
    print(code)

# **********************************************************************************
def run_tests():
    # ...
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, 'scripts')

    files = sorted(os.listdir(path_dir))
    files = [os.path.join(path_dir,f) for f in files if (f.endswith(".py"))]
    # ...

    for fname in files:
        test_py2py(fname)

######################
if __name__ == '__main__':
#    run_tests()
    test_py2py('scripts/py2py/ex0.py')
