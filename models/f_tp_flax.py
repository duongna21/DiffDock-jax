from typing import List, Optional, Iterator

import e3nn_jax as e3nn
from e3nn_jax.legacy import FunctionalTensorProduct
import flax.linen as nn
import jax.numpy as jnp
import jax

def FunctionalFullTensorProduct(
    irreps_in1: e3nn.Irreps,
    irreps_in2: e3nn.Irreps,
    in1_var: Optional[List[float]] = None,
    in2_var: Optional[List[float]] = None,
    out_var: Optional[List[float]] = None,
    filter_ir_out: Iterator[e3nn.Irrep] = None,
    irrep_normalization: str = None,
    path_normalization: str = None,
    gradient_normalization: str = None,
):
    irreps_in1 = e3nn.Irreps(irreps_in1).simplify()
    irreps_in2 = e3nn.Irreps(irreps_in2).simplify()
    
    if filter_ir_out is not None:
        try:
            filter_ir_out = [e3nn.Irrep(ir) for ir in filter_ir_out]
        except ValueError:
            raise ValueError(f"filter_ir_out (={filter_ir_out}) must be an iterable of e3nn.Irrep")

    out = []
    instr = []
    for i_1, (mul_1, ir_1) in enumerate(irreps_in1):
        for i_2, (mul_2, ir_2) in enumerate(irreps_in2):
            for ir_out in ir_1 * ir_2:

                if filter_ir_out is not None and ir_out not in filter_ir_out:
                    continue

                i_out = len(out)
                out.append((mul_1 * mul_2, ir_out))
                instr += [
                    (i_1, i_2, i_out, 'uvuv', False)
                ]

    irreps_out = e3nn.Irreps(out)
    out, p, _ = out.sort()

    instructions = [
        (i_1, i_2, p[i_out], mode, train)
        for i_1, i_2, i_out, mode, train in instr
    ]
       
    return FunctionalTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        in1_var,
        in2_var,
        out_var,
        irrep_normalization,
        path_normalization,
        gradient_normalization,
    )

class FullTensorProduct(nn.Module):
    
    irreps_out: e3nn.Irreps
    irreps_in1: Optional[e3nn.Irreps] = None
    irreps_in2: Optional[e3nn.Irreps] = None

    @nn.compact
    def __call__(
        self, x1: e3nn.IrrepsArray, x2: e3nn.IrrepsArray, **kwargs
    ) -> e3nn.IrrepsArray:
        
        irreps_out = e3nn.Irreps(self.irreps_out)
        irreps_in1 = e3nn.Irreps(self.irreps_in1) if self.irreps_in1 is not None else None
        irreps_in2 = e3nn.Irreps(self.irreps_in2) if self.irreps_in2 is not None else None
        
        x1 = e3nn.as_irreps_array(x1)
        x2 = e3nn.as_irreps_array(x2)

        leading_shape = jnp.broadcast_shapes(x1.shape[:-1], x2.shape[:-1])
        x1 = x1.broadcast_to(leading_shape + (-1,))
        x2 = x2.broadcast_to(leading_shape + (-1,))

        if self.irreps_in1 is not None:
            x1 = x1.rechunk(irreps_in1)
        if self.irreps_in2 is not None:
            x2 = x2.rechunk(irreps_in2)

        x1 = x1.remove_zero_chunks().simplify()
        x2 = x2.remove_zero_chunks().simplify()

        tp = FunctionalFullTensorProduct(
            x1.irreps, x2.irreps, irreps_out.simplify()
        )
        ws = [
            self.param(
                (
                    f"w[{ins.i_in1},{ins.i_in2},{ins.i_out}] "
                    f"{tp.irreps_in1[ins.i_in1]},{tp.irreps_in2[ins.i_in2]},{tp.irreps_out[ins.i_out]}"
                ),
                jax.nn.initializers.normal(stddev=ins.weight_std),
                ins.path_shape
            )
            for ins in tp.instructions
        ]

        f = lambda x1, x2: tp.left_right(ws, x1, x2, **kwargs)

        for _ in range(len(leading_shape)):
            f = e3nn.utils.vmap(f)

        output = f(x1, x2)
        return output.rechunk(irreps_out)