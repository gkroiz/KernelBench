import jax
import jax.numpy as jnp
from flax import nnx
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def matmul_kernel(x_ref, y_ref, z_ref):
    @pl.when(pl.program_id(2) == 0)
    def _():
        z_ref[...] = jnp.zeros_like(z_ref)

    z_ref[...] += x_ref[...] @ y_ref[...]


def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
):
    m, k = x.shape
    _, n = y.shape
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
            pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
        grid=(m // bm, n // bn, k // bk),
        compiler_params=pltpu.TPUCompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
    )(x, y)


class ModelNew(nnx.Module):
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """

    def __call__(self, A: jax.Array, B: jax.Array) -> jax.Array:
        """
        Performs the matrix multiplication.

        Args:
            A (jax.Array): Input matrix A of shape (N, N).
            B (jax.Array): Input matrix B of shape (N, N).

        Returns:
            jax.Array: Output matrix C of shape (N, N).
        """
        return matmul(A, B)
