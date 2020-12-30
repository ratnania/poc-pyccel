# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

# Note that we need to change the directory for tests involving the import
# statement

import os

from pyccel.parser.parser   import Parser
from pyccel.codegen.codegen import Codegen
from pyccel.errors.errors   import Errors

from pyccel.ast.core import Variable
from pyccel.ast.core import For
from pyccel.ast.core import Assign
from pyccel.ast.core import CodeBlock
from pyccel.ast.core import FunctionDef
from pyccel.ast.operators import PyccelFloorDiv
from pyccel.ast.builtins  import PythonInt
from pyccel.ast.builtins  import PythonRange
from pyccel.ast.literals  import LiteralInteger

# **********************************************************************************
def _extract_loop(expr, index):
    if not isinstance(expr, FunctionDef):
        raise TypeError('expr must be a FunctionDef.')

    if not isinstance(index, str):
        raise TypeError('index must be a string.')

    body = expr.body
    # body is a CodeBlock

    for expr in body.body:
        if isinstance(expr, For):
            if expr.target.name == index:
                return expr

    raise ValueError('expr does not have any loop with target = {}'.format(index))


# **********************************************************************************
def split(expr, index, size):
    if isinstance(expr, FunctionDef):
        f = expr
        body = split(f.body, index, size)
        return FunctionDef( f.name,
                            f.arguments,
                            f.results,
                            body,
                            local_vars=f.local_vars,
                            global_vars=f.global_vars,
                            imports=f.imports,
                            decorators=f.decorators,
                            headers=f.headers,
                            templates=f.templates,
                            is_recursive=f.is_recursive,
                            is_pure=f.is_pure,
                            is_elemental=f.is_elemental,
                            is_private=f.is_private,
                            is_header=f.is_header,
                            arguments_inout=f.arguments_inout,
                            functions=f.functions,
                            interfaces=f.interfaces,
                            doc_string=f.doc_string )

    elif isinstance(expr, CodeBlock):
        expr = [split(a, index, size) for a in expr.body]
        return CodeBlock(expr)

    elif isinstance(expr, For):
        target = expr.target
        body   = expr.body
        if target.name == index:
            iterable = expr.iterable
            if isinstance(iterable, PythonRange):
                start = iterable.start
                stop  = iterable.stop
                step  = iterable.step

#                if not( step == 1 ):
#                    raise NotImplementedError('Only step = 1 is handled')

                inner = Variable('int', 'inner_{}'.format(target.name))
                outer = Variable('int', 'outer_{}'.format(target.name))

                inner_range = PythonRange(0, size, 1) # TODO what about step?
                body = CodeBlock([Assign(target, inner+size*outer), body])
                inner_loop = For(inner, inner_range, body)

                new_stop = Variable('int', 'stop_{}'.format(outer.name))

                assign_tmp = Assign(new_stop,  size-1+stop)
                assign_stop = Assign(new_stop,  PyccelFloorDiv(new_stop, LiteralInteger(size)) )
                outer_range = PythonRange(start, new_stop, step)
                outer_loop = For(outer, outer_range, [inner_loop])

                body = CodeBlock([assign_tmp, assign_stop, outer_loop])

                return body

            else:
                raise TypeError('Not yet available')

        else:
            return split(expr.body, index, size)

    else:
        raise NotImplementedError('TODO {}'.format(type(expr)))


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
    print(loop)

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
    run_tests()
    test_split_1('scripts/ex1.py')
