from math import prod

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

import e3nn_jax as e3nn


def _batch_norm(
    input,
    running_mean,
    running_var,
    weight,
    bias,
    normalization,
    reduce,
    is_training,
    is_instance,
    has_affine,
    momentum,
    epsilon,
):
    def _roll_avg(curr, update):
        return (1 - momentum) * curr + momentum * jax.lax.stop_gradient(update)

    batch, *size = input.shape[:-1]
    # TODO add test case for when prod(size) == 0

    input = input.reshape((batch, prod(size), -1))

    new_means = []
    new_vars = []

    fields = []

    i_wei = 0  # index for running_var and weight
    i_rmu = 0  # index for running_mean
    i_bia = 0  # index for bias

    for (mul, ir), field in zip(input.irreps, input.chunks):
        if field is None:
            # [batch, sample, mul, repr]
            if ir.is_scalar():  # scalars
                if is_training or is_instance:
                    if not is_instance:
                        new_means.append(jnp.zeros((mul,), dtype=input.dtype))
                i_rmu += mul

            if is_training or is_instance:
                if not is_instance:
                    new_vars.append(jnp.ones((mul,), input.dtype))

            if has_affine and ir.is_scalar():  # scalars
                i_bia += mul

            fields.append(field)  # [batch, sample, mul, repr]
        else:
            # [batch, sample, mul, repr]
            if ir.is_scalar():  # scalars
                if is_training or is_instance:
                    if is_instance:
                        field_mean = field.mean(1).reshape(batch, mul)  # [batch, mul]
                    else:
                        field_mean = field.mean([0, 1]).reshape(mul)  # [mul]
                        new_means.append(
                            _roll_avg(running_mean.value[i_rmu : i_rmu + mul], field_mean)
                        )
                else:
                    field_mean = running_mean.value[i_rmu : i_rmu + mul]
                i_rmu += mul

                # [batch, sample, mul, repr]
                field = field - field_mean.reshape(-1, 1, mul, 1)

            if is_training or is_instance:
                if normalization == "norm":
                    field_norm = jnp.square(field).sum(3)  # [batch, sample, mul]
                elif normalization == "component":
                    field_norm = jnp.square(field).mean(3)  # [batch, sample, mul]
                else:
                    raise ValueError(
                        "Invalid normalization option {}".format(normalization)
                    )

                if reduce == "mean":
                    field_norm = field_norm.mean(1)  # [batch, mul]
                elif reduce == "max":
                    field_norm = field_norm.max(1)  # [batch, mul]
                else:
                    raise ValueError("Invalid reduce option {}".format(reduce))

                if not is_instance:
                    field_norm = field_norm.mean(0)  # [mul]
                    new_vars.append(
                        _roll_avg(running_var.value[i_wei : i_wei + mul], field_norm)
                    )
            else:
                field_norm = running_var.value[i_wei : i_wei + mul]

            field_norm = jax.lax.rsqrt(
                (1 - epsilon) * field_norm + epsilon
            )  # [(batch,) mul]

            if has_affine:
                sub_weight = weight[i_wei : i_wei + mul]  # [mul]
                field_norm = field_norm * sub_weight  # [(batch,) mul]

            # TODO add test case for when mul == 0
            field_norm = field_norm[..., None, :, None]  # [(batch,) 1, mul, 1]
            field = field * field_norm  # [batch, sample, mul, repr]

            if has_affine and ir.is_scalar():  # scalars
                sub_bias = bias[i_bia : i_bia + mul]  # [mul]
                field += sub_bias.reshape(mul, 1)  # [batch, sample, mul, repr]
                i_bia += mul

            fields.append(field)  # [batch, sample, mul, repr]
        i_wei += mul

    output = e3nn.from_chunks(input.irreps, fields, (batch, prod(size)), input.dtype)
    output = output.reshape((batch,) + tuple(size) + (-1,))
    return output, new_means, new_vars




class BatchNorm(nn.Module):
    """Equivariant Batch Normalization.

    It normalizes by the norm of the representations.
    Note that the norm is invariant only for orthonormal representations.
    Irreducible representations are orthonormal.

    Args:
        irreps: Irreducible representations of the input and output (unchanged)
        eps (float): epsilon for numerical stability, has to be between 0 and 1.
            the field norm is transformed to ``(1 - eps) * norm + eps``
            leading to a slower convergence toward norm 1.
        momentum: momentum for moving average
        affine: whether to include learnable biases
        reduce: reduce mode, either 'mean' or 'max'
        instance: whether to use instance normalization
        normalization: normalization mode, either 'norm' or 'component'
    """
    irreps: e3nn.Irreps = None
    eps: float = 1e-4
    momentum: float = 0.1
    affine: bool = True
    reduce: str = "mean"
    instance: bool = False
    normalization: str = None

    def __repr__(self):
        return f"{self.__class__.__name__} ({e3nn.Irreps(self.irreps)}, eps={self.eps}, momentum={self.momentum})"

    @nn.compact
    def __call__(
        self, input: e3nn.IrrepsArray, is_training: bool = True
    ) -> e3nn.IrrepsArray:
        r"""Evaluate the batch normalization.

        Args:
            input: input tensor of shape ``(batch, [spatial], irreps.dim)``
            is_training: whether to train or evaluate

        Returns:
            output: normalized tensor of shape ``(batch, [spatial], irreps.dim)``
        """
        assert isinstance(self.reduce, str), "reduce should be passed as a string value"
        assert self.reduce in ["mean", "max"], "reduce needs to be 'mean' or 'max'"
        norm = self.normalization
        f_irreps = e3nn.Irreps(self.irreps)
        if self.normalization is None:
            norm = e3nn.config("irrep_normalization")
        assert norm in [
            "norm",
            "component",
        ], "normalization needs to be 'norm' or 'component'"
        
        
        if f_irreps is not None:
            input = input.rechunk(f_irreps)

        num_scalar = sum(mul for mul, ir in input.irreps if ir.is_scalar())
        num_features = input.irreps.num_irreps

        if not self.instance:
            running_mean = self.variable(
                "batch_stats", "running_mean", jnp.zeros, (num_scalar,)
            )
            running_var = self.variable(
                "batch_stats", "running_var", jnp.ones, (num_features,)
            )
        else:
            running_mean = None
            running_var = None

        if self.affine:
            weight = self.param("weight", nn.initializers.ones, (num_features,))
            bias = self.param("bias", nn.initializers.zeros, (num_scalar,))
        else:
            weight = None
            bias = None

        output, new_means, new_vars = _batch_norm(
            input,
            running_mean=running_mean,
            running_var=running_var,
            weight=weight,
            bias=bias,
            normalization=norm,
            reduce=self.reduce,
            is_training=is_training,
            is_instance=self.instance,
            has_affine=self.affine,
            momentum=self.momentum,
            epsilon=self.eps,
        )
        
        if is_training and not self.instance:
            if len(new_means):
                running_mean.value = jnp.concatenate(new_means)
            if len(new_vars):
                running_var.value = jnp.concatenate(new_vars)

        return output
