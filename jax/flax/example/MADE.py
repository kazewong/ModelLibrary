from typing import Callable, Sequence
import jax 
import jax.numpy as jnp
from jax import random
from flax import linen as nn
import optax

def get_masks(input_dim, hidden_dim=64, num_hidden=1):
    masks = []
    input_degrees = jnp.arange(input_dim)
    degrees = [input_degrees]

    for n_h in range(num_hidden + 1):
        degrees += [jnp.arange(hidden_dim) % (input_dim - 1)]
    degrees += [input_degrees % input_dim - 1]

    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        masks += [jnp.transpose(jnp.expand_dims(d1, -1) >= jnp.expand_dims(d0, 0)).astype(jnp.float32)]
    return masks

class MaskedDense(nn.Module):
    n_dim: int
    n_hidden: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x, mask):
        weight = self.param('weights', self.kernel_init, (self.n_dim, self.n_hidden))
        bias = self.param('bias', self.bias_init, (self.n_hidden,))
        return jnp.dot(x, weight * mask) + bias

class MaskedAutoEncoder(nn.Module):
    n_dim: int
    n_hidden: int

    def setup(self):
        self.mask = get_masks(self.n_dim, self.n_hidden)
        self.up = MaskedDense(self.n_dim, self.n_hidden)
        self.mid = MaskedDense(self.n_hidden, self.n_hidden)
        self.down = MaskedDense(self.n_hidden, 2*self.n_dim)

    def __call__(self, inputs):
        log_weight, bias = self.one_pass(inputs)
        outputs = (inputs - bias)*jnp.exp(-log_weight)
        log_jacobian = -jnp.sum(log_weight, axis=-1)
        return outputs, log_jacobian

    def one_pass(self, inputs):
        x = self.up(inputs, self.mask[0])
        x = nn.swish(x)
        x = self.mid(x, self.mask[1])
        x = nn.swish(x)
        log_weight, bias = self.down(x, self.mask[2].tile(2)).split(2, -1)
        return log_weight, bias

    def inverse(self, inputs):
        outputs = jnp.zeros_like(inputs)
        for i_col in range(inputs.shape[1]):
            log_weight, bias = self.one_pass(outputs)
            outputs = jax.ops.index_update(
                outputs, jax.ops.index[:, i_col], inputs[:, i_col] * jnp.exp(log_weight[:, i_col]) + bias[:, i_col]
            )
        log_det_jacobian = -log_weight.sum(-1)
        return outputs, log_det_jacobian