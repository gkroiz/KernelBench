import jax
from flax import nnx
from jax.experimental import pallas as pl


def add_vectors_kernel(x_ref, y_ref, o_ref):
    x, y = x_ref[...], y_ref[...]
    o_ref[...] = x + y


def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
    return pl.pallas_call(
        add_vectors_kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
    )(x, y)


class ModelNew(nnx.Module):
    def __call__(self, A: jax.Array, B: jax.Array) -> jax.Array:
        return add_vectors(A, B)
