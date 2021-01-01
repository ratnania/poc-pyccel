# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

# Note that we need to change the directory for tests involving the import
# statement

import os

from sympy import Tuple
from sympy.core.expr          import Expr

from pyccel.parser.parser   import Parser
from pyccel.codegen.codegen import Codegen
from pyccel.errors.errors   import Errors

from pyccel.ast.core import Variable
from pyccel.ast.core import For
from pyccel.ast.core import Assign
from pyccel.ast.core import While
from pyccel.ast.core import If
from pyccel.ast.core import Return
from pyccel.ast.core import Assign
from pyccel.ast.core import CodeBlock
from pyccel.ast.core import FunctionDef
from pyccel.ast.operators import PyccelFloorDiv
from pyccel.ast.builtins  import PythonInt
from pyccel.ast.builtins  import PythonRange
from pyccel.ast.literals  import LiteralInteger
from pyccel.codegen.printing.pycode import pycode

# **********************************************************************************
# TODO works with one variable for the moment
def _subs(expr, old, new):
    if isinstance(expr, CodeBlock):
        body = []
        for stmt in expr.body:
            new_stmt = _subs(stmt, old, new)
            body.append(new_stmt)

        return CodeBlock(body)

    elif isinstance(expr, For):
        body = _subs(expr.body, old, new)
        return For(expr.target, expr.iterable, body)

    else:
        return expr.subs(old, new)

# **********************************************************************************
def _extract_loop(expr, index):
    if isinstance(expr, FunctionDef):
        expr = expr.body
        # expr is a CodeBlock

#    else:
#        print(type(expr))
#        import sys; sys.exit(0)

    if not isinstance(index, str):
        raise TypeError('index must be a string.')

    for e in expr.body:
        if isinstance(e, For):
            if e.target.name == index:
                return e
            else:
                return _extract_loop(e.body, index)

    raise ValueError('expr does not have any loop with target = {}'.format(index))


# **********************************************************************************
def _split_For(expr, index, size, inner_unroll=False):
    if not (expr.target.name == index):
        body = split(expr.body, index, size, inner_unroll=inner_unroll)
        return For(expr.target, expr.iterable, body)

    target = expr.target
    body   = expr.body
    iterable = expr.iterable
    if isinstance(iterable, PythonRange):
        # TODO use the same code as in unroll
        start = iterable.start
        stop  = iterable.stop
        step  = iterable.step

        if not( step.python_value == 1 ):
            raise NotImplementedError('Only step = 1 is handled')

        inner = Variable('int', 'inner_{}'.format(target.name))
        outer = Variable('int', 'outer_{}'.format(target.name))

        inner_range = PythonRange(0, size, 1) # TODO what about step?
        body = _subs(body, target, inner+size*outer)
        inner_loop = For(inner, inner_range, body)

        if inner_unroll:
            inner_loop = unroll(inner_loop)
        else:
            inner_loop = [inner_loop]

        new_stop = Variable('int', 'stop_{}'.format(outer.name))

        assign_tmp = Assign(new_stop,  size-1+stop)
        assign_stop = Assign(new_stop,  PyccelFloorDiv(new_stop, LiteralInteger(size)) )
        outer_range = PythonRange(start, new_stop, step)
        outer_loop = For(outer, outer_range, inner_loop)

        body = CodeBlock([assign_tmp, assign_stop, outer_loop])

        return body

    else:
        raise TypeError('Not yet available')

# **********************************************************************************
def _split_CodeBlock(expr, index, size, inner_unroll=False):
    body = []
    for stmt in expr.body:
        new = split(stmt, index, size, inner_unroll=inner_unroll)
        body.append(new)

    return CodeBlock(body)

# **********************************************************************************
def split(expr, index, size, inner_unroll=False):
    if isinstance(expr, For):
        return _split_For(expr, index, size, inner_unroll=inner_unroll)

    elif isinstance(expr, CodeBlock):
        return _split_CodeBlock(expr, index, size, inner_unroll=inner_unroll)

    elif isinstance(expr, Assign):
        return expr

    else:
        raise TypeError('Not available for {}'.format(type(expr)))

# **********************************************************************************
def unroll(expr):
    iterable = expr.iterable
    target = expr.target

    if not isinstance(iterable, PythonRange):
        raise TypeError('Expecting PythonRange')

    start = iterable.start
    stop  = iterable.stop
    step  = iterable.step

    # ...
    if isinstance(start, LiteralInteger):
        start = start.python_value

    elif not isinstance(start, int):
        raise TypeError('Expecting LiteralInteger or int')
    # ...

    # ...
    if isinstance(step, LiteralInteger):
        step = step.python_value

    elif not isinstance(step, int):
        raise TypeError('Expecting LiteralInteger or int')
    # ...

    # ...
    if isinstance(stop, LiteralInteger):
        stop = stop.python_value

    elif not isinstance(stop, int):
        raise TypeError('Expecting LiteralInteger or int')
    # ...

    stmts = expr.body.body # this is a CodeBlock

    body = []
    for stmt in stmts:
        for i in range(start, stop, step):
            new = stmt.subs(target, target+i)
            body.append(new)

    body = CodeBlock(body)
    return body

# **********************************************************************************
class SplitLoop(object):
    def __init__(self, index, size):
        self._index = index
        self._size = size

    @property
    def index(self):
        return self._index

    @property
    def size(self):
        return self._size

# **********************************************************************************
class Transform(object):
    def __init__(self, filename):
        self._filename = filename

        # ... run the semantic stage
        pyccel = Parser(filename)
        ast = pyccel.parse()

        settings = {}
        ast = pyccel.annotate(**settings)

        name = os.path.basename(filename)
        name = os.path.splitext(name)[0]

        codegen = Codegen(ast, name)
        if not( len( codegen.expr.funcs ) == 1 ):
            raise ValueError('Expecting one single function')

        self._func = codegen.expr.funcs[0]
        self._codegen = codegen

        # reset Errors singleton
        errors = Errors()
        errors.reset()

    @property
    def codegen(self):
        return self._codegen

    @property
    def func(self):
        return self._func

    def update(self, func):
        self._codegen.expr.funcs[0] = func

    def extract_loop(self, index):
        return _extract_loop(self.func, index)

    def doprint(self, language='python'):
        return self.codegen.doprint(language=language)

# **********************************************************************************
def transform(fname, **kwargs):
    # ...
    T = Transform(fname)
    f = T.func
    # ...

    # ...
    transformations = kwargs.pop('transformations', [])
    for transform in transformations:
        if isinstance(transform, SplitLoop):
            index = transform.index
            size = transform.size
            f = split(f, index=index, size=size)

            T.update(f)
    # ...

    return T.doprint()

# **********************************************************************************
def test_split_1(fname, **kwargs):
    T = Transform(fname)
    loop = T.extract_loop(index='i')

    print('****************** BEFORE ******************')
    code = pycode(loop)
    print(code)

    print('****************** AFTER  ******************')
    loop = split(loop, 'i', 16)
    code = pycode(loop)
    print(code)

# **********************************************************************************
def test_split_2(fname, **kwargs):
    T = Transform(fname)
    loop = T.extract_loop(index='i')

    print('****************** BEFORE ******************')
    code = pycode(loop)
    print(code)

    print('****************** SPLIT i *****************')
    loop = split(loop, 'i', 8)
    code = pycode(loop)
    print(code)
    print('****************** SPLIT j *****************')
    loop = split(loop, 'j', 4)
    code = pycode(loop)
    print(code)

# **********************************************************************************
def test_split_unroll_1(fname, **kwargs):
    T = Transform(fname)
    loop = T.extract_loop(index='i')

    print('****************** BEFORE ******************')
    code = pycode(loop)
    print(code)

    print('****************** AFTER  ******************')
    loop = split(loop, 'i', 4, inner_unroll=True)
    code = pycode(loop)
    print(code)

# **********************************************************************************
def test_unroll_1(fname, **kwargs):
    T = Transform(fname)
    loop = T.extract_loop(index='i')

    print('****************** BEFORE ******************')
    code = pycode(loop)
    print(code)

    print('****************** AFTER  ******************')
    loop = unroll(loop)
    code = pycode(loop)
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
        code = transform(fname, transformations=[SplitLoop('i', 16)])
        print('****************** AFTER  ******************')
        print(code)

######################
if __name__ == '__main__':
#    run_tests()
#    test_split_1('scripts/ex1.py')
    test_split_2('scripts/ex3.py')
#    test_split_unroll_1('scripts/ex1.py')
#    test_unroll_1('scripts/ex2.py')
