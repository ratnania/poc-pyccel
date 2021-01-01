# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

# Note that we need to change the directory for tests involving the import
# statement

import os
from collections import OrderedDict

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
from pyccel.ast.core import EmptyNode
from pyccel.ast.operators import PyccelFloorDiv
from pyccel.ast.builtins  import PythonInt
from pyccel.ast.builtins  import PythonRange
from pyccel.ast.literals  import LiteralInteger

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
        body = _split(expr.body, index, size, inner_unroll=inner_unroll)
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
        new = _split(stmt, index, size, inner_unroll=inner_unroll)
        body.append(new)

    return CodeBlock(body)

# **********************************************************************************
def _split_FunctionDef(expr, index, size, inner_unroll=False):
    f = expr
    body = _split(f.body, index, size, inner_unroll=inner_unroll)
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

# **********************************************************************************
def _split(expr, index, size, inner_unroll=False):
    if isinstance(expr, For):
        return _split_For(expr, index, size, inner_unroll=inner_unroll)

    elif isinstance(expr, CodeBlock):
        return _split_CodeBlock(expr, index, size, inner_unroll=inner_unroll)

    elif isinstance(expr, FunctionDef):
        return _split_FunctionDef(expr, index, size, inner_unroll=inner_unroll)

    elif isinstance(expr, (Assign, EmptyNode)):
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

    # ...
    def _subs_index(stmt, old, new):
        if isinstance(stmt, CodeBlock):
            body = []
            for i in stmt.body:
                body.append(_subs_index(i, old, new))

            return CodeBlock(body)

        elif isinstance(stmt, For):
            body = _subs_index(stmt.body, old, new)

            return For(stmt.target, stmt.iterable, body)

        else:
            return stmt.subs(old, new)
    # ...

    body = []
    for stmt in stmts:
        for i in range(start, stop, step):
            new = _subs_index(stmt, target, i)
            body.append(new)

    body = CodeBlock(body)
    return body


# **********************************************************************************
def reorder(expr, *args):
    return expr

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

    def split(self, *args):
        for a in args:
            assert(isinstance(a, (dict, OrderedDict)))

        expr = self.func
        for d in args:
            index        = d['index']
            size         = d['size']
            inner_unroll = d['inner_unroll']

            expr = _split(expr, index, size, inner_unroll=inner_unroll)

        return expr
