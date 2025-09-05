from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs


def _var(name, shape=None):
    return gs.Variable(name=name, dtype=np.float32, shape=shape)


def _const(name, value):
    arr = np.array(value, dtype=np.float32)
    return gs.Constant(name=name, values=arr)


def replace_all_layernorms(graph: gs.Graph) -> int:
    replaced = 0
    new_nodes = []

    for ln in list(graph.nodes):
        if ln.op != "LayerNormalization":
            continue

        X = ln.inputs[0]
        gamma = ln.inputs[1] if len(ln.inputs) >= 2 else None
        beta = ln.inputs[2] if len(ln.inputs) >= 3 else None

        axis = ln.attrs["axis"]
        epsilon = float(ln.attrs["epsilon"])

        dtype = np.float32

        base = ln.name

        mean = _var(f"{base}/mean")
        n_mean = gs.Node(
            op="ReduceMean",
            inputs=[X],
            outputs=[mean],
            attrs={"axes": [int(axis)]},
            name=f"{base}/ReduceMean",
        )

        xc = _var(f"{base}/xc")
        n_sub = gs.Node("Sub", inputs=[X, mean], outputs=[xc], name=f"{base}/Sub")

        pow_const = _const(f"{base}/Constant_output_0", 2.0)
        sq = _var(f"{base}/sq")
        n_pow = gs.Node("Pow", inputs=[xc, pow_const], outputs=[sq], name=f"{base}/Pow")

        var = _var(f"{base}/var")
        n_mean2 = gs.Node(
            op="ReduceMean",
            inputs=[sq],
            outputs=[var],
            attrs={"axes": [int(axis)]},
            name=f"{base}/ReduceMean_1",
        )

        eps_c = _const(f"{base}/epsilon", epsilon)
        var_eps = _var(f"{base}/var_plus_eps")
        n_add_eps = gs.Node(
            "Add", inputs=[var, eps_c], outputs=[var_eps], name=f"{base}/Add_epsilon"
        )

        denom = _var(f"{base}/denom")
        n_sqrt = gs.Node("Sqrt", inputs=[var_eps], outputs=[denom], name=f"{base}/Sqrt")

        norm = _var(f"{base}/norm")
        n_div = gs.Node("Div", inputs=[xc, denom], outputs=[norm], name=f"{base}/Div")

        last = norm
        n_mul = None
        if gamma is not None:
            scaled = _var(f"{base}/scaled")
            n_mul = gs.Node(
                "Mul", inputs=[last, gamma], outputs=[scaled], name=f"{base}/Mul_gamma"
            )
            last = scaled

        n_add_beta = None
        if beta is not None:
            shifted = _var(f"{base}/shifted")
            n_add_beta = gs.Node(
                "Add", inputs=[last, beta], outputs=[shifted], name=f"{base}/Add_beta"
            )
            last = shifted

        Y_old = ln.outputs[0]
        graph.outputs = [last if o is Y_old else o for o in graph.outputs]
        for consumer in list(Y_old.outputs):
            consumer.inputs = [last if inp is Y_old else inp for inp in consumer.inputs]

        new_nodes.extend([n_mean, n_sub, n_pow, n_mean2, n_add_eps, n_sqrt, n_div])
        if n_mul is not None:
            new_nodes.append(n_mul)
        if n_add_beta is not None:
            new_nodes.append(n_add_beta)

        graph.nodes.remove(ln)
        replaced += 1

    graph.nodes += new_nodes
    return replaced
