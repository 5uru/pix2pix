from typing import List

from jax  import numpy as jnp
from flax import linen as nn
import jax

class Discriminator(nn.Module):
    """Discriminator network."""

    @nn.compact
    def __call__(self, x, y):

        x = jnp.concatenate([x, y], axis=1)
        block1 = nn.Conv( 16, 3, 2, padding='SAME')(y)
        block1 = nn.BatchNorm(use_running_average=False)(block1)
        block1 = nn.activation.leaky_relu(block1, 0.2)
        print(block1.shape)




def test_discriminator():
    model = Discriminator()
    satellite_image = jnp.ones((1,1,28,28))
    map_image = jnp.ones((1,1,28,28))

    params = model.init(jax.random.PRNGKey(0), satellite_image, map_image)
    y = model.apply(params, satellite_image, map_image, mutable=['batch_stats'])


test_discriminator()