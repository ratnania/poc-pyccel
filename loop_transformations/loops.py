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

from pyccel.ast.basic     import Basic
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
class OldSplitFor(Basic):
    def __new__(cls, loop, size, inner_unroll):
        # TODO we should not do this all the time

        assert(isinstance(loop, For))

        if not isinstance(loop.iterable, PythonRange):
            raise TypeError('iterable must be of type Range')

        target = loop.target
        body   = loop.body
        iterable = loop.iterable

        # TODO use the same code as in unroll
        start = iterable.start
        stop  = iterable.stop
        step  = iterable.step

        if not( step.python_value == 1 ):
            raise NotImplementedError('Only step = 1 is handled')

        inner = Variable('int', 'inner_{}'.format(target.name))
        outer = Variable('int', 'outer_{}'.format(target.name))

        # ...
        inner_range = PythonRange(0, size, 1) # TODO what about step?
        body = _subs(body, target, inner+size*outer)
        inner_loop = For(inner, inner_range, body)

        if inner_unroll:
            inner_loop = unroll(inner_loop)

        inner_loop = OldInnerFor(inner_loop)
        # ...

        # ...
        new_stop = Variable('int', 'stop_{}'.format(outer.name))

        assign_tmp = Assign(new_stop,  size-1+stop)
        assign_stop = Assign(new_stop,  PyccelFloorDiv(new_stop, LiteralInteger(size)) )
        outer_range = PythonRange(start, new_stop, step)
        outer_loop = For(outer, outer_range, [inner_loop])

        body = CodeBlock([assign_tmp, assign_stop, outer_loop])

        outer_loop = OldOuterFor(body)
        # ...

        return Basic.__new__(cls, loop, outer_loop, inner_loop, size, inner_unroll)

    @property
    def loop(self):
        return self._args[0]

    @property
    def outer(self):
        return self._args[1]

    @property
    def inner(self):
        return self._args[2]

    @property
    def size(self):
        return self._args[3]

    @property
    def inner_unroll(self):
        return self._args[4]

# **********************************************************************************
class InnerFor(Basic):
    def __new__(cls, target, size, unroll, body):
        iterable = PythonRange(0, size, 1) # TODO what about step?
        return Basic.__new__(cls, target, iterable, size, unroll, body)

    @property
    def target(self):
        return self._args[0]

    @property
    def iterable(self):
        return self._args[1]

    @property
    def size(self):
        return self._args[2]

    @property
    def unroll(self):
        return self._args[3]

    @property
    def body(self):
        return self._args[4]

# **********************************************************************************
class OuterFor(Basic):
    def __new__(cls, target, iterable, inner):
        # ...
        if not isinstance(iterable, PythonRange):
            raise TypeError('iterable must be of type Range')

        start = iterable.start
        stop  = iterable.stop
        step  = iterable.step

        if not( step.python_value == 1 ):
            raise NotImplementedError('Only step = 1 is handled')
        # ...

        stop_var = Variable('int', 'stop_{}'.format(target.name))

        # ... TODO improve
        size = inner.size
        assign_tmp = Assign(stop_var,  size-1+stop)
        assign_stop = Assign(stop_var,  PyccelFloorDiv(stop_var, LiteralInteger(size)) )

        prelude = CodeBlock([assign_tmp, assign_stop])
        # ...

        iterable = PythonRange(start, stop_var, step)

        return Basic.__new__(cls, target, iterable, prelude, inner)

    @property
    def target(self):
        return self._args[0]

    @property
    def iterable(self):
        return self._args[1]

    @property
    def prelude(self):
        return self._args[2]

    @property
    def inner(self):
        return self._args[3]

# **********************************************************************************
class SplittedFor(Basic):
    def __new__(cls, outer, inner):
        return Basic.__new__(cls, outer, inner)

    @property
    def outer(self):
        return self._args[0]

    @property
    def inner(self):
        return self._args[1]

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

        self._loops = OrderedDict()
        self._inner_indices = OrderedDict()
        self._outer_indices = OrderedDict()
        self._outer_loops = OrderedDict()
        self._indices = OrderedDict()
        self._new_indices = OrderedDict()

        # reset Errors singleton
        errors = Errors()
        errors.reset()

    @property
    def codegen(self):
        return self._codegen

    @property
    def func(self):
        return self._func

    @property
    def indices(self):
        return self._indices

    @property
    def new_indices(self):
        return self._new_indices

    @property
    def inner_indices(self):
        return self._inner_indices

    @property
    def outer_indices(self):
        return self._outer_indices

    @property
    def outer_loops(self):
        return self._outer_loops

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

        # ... create inner and outer indices
        self._indices = OrderedDict()
        self._new_indices = OrderedDict()
        for d in args:
            index  = d['index']
            size   = d['size']
            unroll = d['inner_unroll']

            self._indices[index] = {'size': size, 'unroll': unroll}

            name = '{}'.format(index)

            inner = Variable('int', 'inner_{}'.format(name))
            outer = Variable('int', 'outer_{}'.format(name))

            self._inner_indices[index] = inner
            self._outer_indices[index] = outer
            self._new_indices[index]   = inner + size * outer
        # ...

        expr = self._split(expr)

        return expr

    def _split(self, expr, **settings):

        classes = type(expr).__mro__
        for cls in classes:
            method = '_split_' + cls.__name__
            if hasattr(self, method):
                obj = getattr(self, method)(expr, **settings)
                return obj
            else:
                raise NotImplementedError('{} not available'.format(method))

    def _split_Assign(self, expr, **settings):
        for old, new in self.new_indices.items():
            # old is a string
            # TODO shall we improve this?
            old = Variable('int', old)

            expr = expr.subs(old, new)

        return expr

    def _split_EmptyNode(self, expr, **settings):
        return expr

    def _split_For(self, expr, **settings):
        body = self._split(expr.body, **settings)

        if expr.target.name in self.indices.keys():

            target = expr.target
            name = '{}'.format(target.name)

            size   = self.indices[name]['size']
            unroll = self.indices[name]['unroll']

            inner_target = self.inner_indices[name]
            outer_target = self.outer_indices[name]

            inner = InnerFor(inner_target, size, unroll, body)
            outer = OuterFor(outer_target, expr.iterable, inner)

            return SplittedFor(outer, inner)

        else:
            return For(expr.target, expr.iterable, body)

    def _split_CodeBlock(self, expr, **settings):
        body = []
        for stmt in expr.body:
            new = self._split(stmt, **settings)
            body.append(new)

        return CodeBlock(body)

    def _split_FunctionDef(self, expr, **settings):
        f = expr
        body = self._split(f.body, **settings)
        body = self._finalize(body, **settings)

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

    def _finalize(self, expr, **settings):

        classes = type(expr).__mro__
        for cls in classes:
            method = '_finalize_' + cls.__name__
            if hasattr(self, method):
                obj = getattr(self, method)(expr, **settings)
                return obj
            else:
                raise NotImplementedError('{} not available'.format(method))

    def _finalize_Assign(self, expr, **settings):
        return expr

    def _finalize_EmptyNode(self, expr, **settings):
        return expr

    def _finalize_For(self, expr, **settings):
        return expr

    def _finalize_CodeBlock(self, expr, **settings):
        body = []
        for stmt in expr.body:
            new = self._finalize(stmt, **settings)
            body.append(new)

        return CodeBlock(body)

    def _finalize_SplittedFor(self, expr, **settings):
#        inner = self._finalize(expr.inner, **settings)
        outer = self._finalize(expr.outer, **settings)

#        self._outer_loops[name] = outer

        # TODO
#        return inner
        return outer

    def _finalize_InnerFor(self, expr, **settings):
        if expr.unroll:
            raise NotImplementedError('unroll = True not available')

        else:
            body = self._finalize(expr.body, **settings)
            return For(expr.target, expr.iterable, body)

    def _finalize_OuterFor(self, expr, **settings):
        body = self._finalize(expr.inner, **settings)
        # TODO add prelude
        if not isinstance(body, CodeBlock):
            body = CodeBlock([body])

        return For(expr.target, expr.iterable, body)
