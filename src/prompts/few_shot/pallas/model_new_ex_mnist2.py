import jax
import jax.numpy as jnp
from flax import nnx
from jax.experimental import pallas as pl


def max_pool2d_kernel(x_ref, y_ref, *, kernel_size, stride):
    """Pallas kernel for max pooling 2D operation."""
    # Initialize output with negative infinity using the correct shape
    y_ref[...] = jnp.full_like(y_ref[...], -jnp.inf)
    
    # Iterate over the pooling window
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            # Create slices for the pooling window
            h_slice = slice(kh, None, stride)
            w_slice = slice(kw, None, stride)
            
            # Get the current window values
            window_vals = x_ref[:, h_slice, w_slice, :]
            
            # Take maximum with current output
            y_ref[...] = jnp.maximum(y_ref[...], window_vals)


def max_pool2d(x, kernel_size=2, stride=2):
    """Max pooling 2D using Pallas kernel."""
    batch_size, height, width, channels = x.shape
    
    # Calculate output dimensions
    out_height = (height - kernel_size) // stride + 1
    out_width = (width - kernel_size) // stride + 1
    
    # Define output shape
    output_shape = (batch_size, out_height, out_width, channels)
    
    # Create a partial function with the kernel parameters
    kernel_fn = lambda x_ref, y_ref: max_pool2d_kernel(x_ref, y_ref, kernel_size=kernel_size, stride=stride)
    
    # Call the Pallas kernel with single grid element
    return pl.pallas_call(
        kernel_fn,
        out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
        grid=1
    )(x)


class ModelNew(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_features=1, out_features=10, kernel_size=(5, 5), strides=(1, 1), padding="VALID", rngs=rngs)
        self.conv2 = nnx.Conv(in_features=10, out_features=20, kernel_size=(5, 5), strides=(1, 1), padding="VALID", rngs=rngs)
        self.fc1 = nnx.Linear(in_features=320, out_features=50, rngs=rngs)
        self.fc2 = nnx.Linear(in_features=50, out_features=10, rngs=rngs)

        self.max_pool2d = max_pool2d

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.max_pool2d(nnx.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = self.max_pool2d(nnx.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = jnp.reshape(x, (-1, 320))
        x = nnx.relu(self.fc1(x))  # Flatten
        x = self.fc2(x)
        return nnx.log_softmax(x)
