import itertools

import graphviz
import jax
import jax._src.source_info_util
import jax.experimental
import jax.experimental.stax
import jax.scipy.special
from graphviz import Digraph
from jax import core


def hlo_graph(f, *args, **kwargs):
    comp = jax.xla_computation(f)(*args, **kwargs)
    graph = graphviz.Source(comp.as_hlo_dot_graph())
    return graph


styles = {
    'const': dict(style='filled', color='goldenrod1'),
    'invar': dict(color='mediumspringgreen', style='filled'),
    'outvar': dict(style='filled,dashed', fillcolor='indianred1', color='black'),
    'op_node': dict(shape='box', color='lightskyblue', style='filled'),
    'intermediate': dict(style='filled', color='cornflowerblue')
}


def _jaxpr_graph(jaxpr):
    id_names = (f'id{id}' for id in itertools.count())
    graph = Digraph(engine='dot')
    graph.attr(size='6,10!')
    for v in jaxpr.constvars:
        graph.node(str(v), core.raise_to_shaped(v.aval).str_short(), styles['const'])
    for v in jaxpr.invars:
        graph.node(str(v), v.aval.str_short(), styles['invar'])
    for eqn in jaxpr.eqns:
        for f in eqn.source_info.frames:
            if "jax/_src" not in f.file_name and "site-packages/jax/" not in f.file_name and "src/jax" not in f.file_name:
                function_name = f"{f.file_name.split('/')[-1]}:{f.function_name}:{f.line_num}"
                break

        for v in eqn.invars:
            if isinstance(v, core.Literal):
                graph.node(
                    str(id(v.val)),
                    core.raise_to_shaped(core.get_aval(v.val)).str_short() + f"\n{v.val}",
                    styles['const']
                )
        if eqn.primitive.multiple_results:
            id_name = next(id_names)
            graph.node(id_name, str(eqn.primitive) + "\n" + function_name, styles['op_node'])
            for v in eqn.invars:
                # graph.edge(str(id(v.val) if isinstance(v, core.Literal) else v), repr(v), id_name)
                graph.edge(str(id(v.val) if isinstance(v, core.Literal) else v), id_name)
            for v in eqn.outvars:
                graph.node(
                    str(v),
                    v.aval.str_short() + f"\n{repr(v)}",
                    styles['intermediate']
                )
                graph.edge(id_name, str(v))
        else:
            outv, = eqn.outvars
            graph.node(
                str(outv),
                str(eqn.primitive) + "\n" + function_name,
                styles['op_node']
            )
            for v in eqn.invars:
                # graph.edge(str(id(v.val) if isinstance(v, core.Literal) else v), str(outv), repr(v))
                graph.edge(str(id(v.val) if isinstance(v, core.Literal) else v), str(outv))
    for i, v in enumerate(jaxpr.outvars):
        outv = 'out_' + str(i)
        graph.node(
            outv,
            outv + f"\n{repr(v)}",
            styles['outvar']
        )
        graph.edge(str(v), outv)
    return graph


def jaxpr_graph(fun, *args):
    jaxpr = jax.make_jaxpr(fun)(*args).jaxpr
    return _jaxpr_graph(jaxpr)


def grad_graph(fun, *args):
    _, fun_vjp = jax.vjp(fun, *args)
    jaxpr = fun_vjp.args[0].func.args[1]
    return _jaxpr_graph(jaxpr)


def plot(fn, x0=None):
    jaxpr_graph(fn, x0).view()
