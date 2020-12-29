# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

# Note that we need to change the directory for tests involving the import
# statement

import os

from pyccel.parser.parser   import Parser
from pyccel.codegen.codegen import Codegen
from pyccel.errors.errors   import Errors

# **********************************************************************************
base_dir = os.path.dirname(os.path.realpath(__file__))
path_dir = os.path.join(base_dir, 'scripts')

files = sorted(os.listdir(path_dir))
files = [os.path.join(path_dir,f) for f in files if (f.endswith(".py"))]

# **********************************************************************************
from pyccel.ast.core import Variable
from pyccel.ast.core import For
from pyccel.ast.core import Assign
from pyccel.ast.core import CodeBlock
from pyccel.ast.core import FunctionDef
from pyccel.ast.operators import PyccelFloorDiv
from pyccel.ast.builtins  import PythonRange


# **********************************************************************************
def split(expr, index, size):
    if isinstance(expr, For):
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

#                assign_stop = Assign(new_stop,  PyccelFloorDiv(size-1+stop, size) ) # TODO not working
                assign_stop = Assign(new_stop, (size-1+stop) / size)
                outer_range = PythonRange(start, new_stop, step)
                outer_loop = For(outer, outer_range, [inner_loop])

                body = CodeBlock([assign_stop, outer_loop])

                return body

            else:
                raise TypeError('Not yet available')

        else:
            return split(expr.body, index, size)

    else:
        raise NotImplementedError('TODO {}'.format(type(expr)))


# **********************************************************************************
def test_codegen(f):

    pyccel = Parser(f)
    ast = pyccel.parse()

    settings = {}
    ast = pyccel.annotate(**settings)

    name = os.path.basename(f)
    name = os.path.splitext(name)[0]

    codegen = Codegen(ast, name)
    f = codegen.expr.funcs[0]
    print('==================')
    loop = f.body.body[0]
    new_loop = split(loop, index='i', size=16)
    new_body = [new_loop]
    fnew =  FunctionDef('{}_new'.format(f.name),
                        f.arguments,
                        f.results,
                        new_body,
                        local_vars=f.local_vars,
                        global_vars=f.global_vars,
#        cls_name=None,
#        is_static=False,
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
        doc_string=f.doc_string
                       )

    codegen.expr.funcs.append(fnew)
    print('==================')

    code = codegen.doprint(language='python')

    # reset Errors singleton
    errors = Errors()
    errors.reset()

    return code
######################
if __name__ == '__main__':
    for f in files:
#        print('> testing {0}'.format(str(os.path.basename(f))))
        code = test_codegen(f)
        print(code)
