# **********************************************************************************
# TODO must work on For only
#      move the rest of the code to the Transform class
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
